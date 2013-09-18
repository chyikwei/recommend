"""
Reference paper: "Bayesian Probabilistic Matrix Factorization using MCMC"
                 R. Salakhutdinov and A.Mnih.  
                 25th International Conference on Machine Learning (ICML-2008) 
Reference Matlab code: http://www.cs.toronto.edu/~rsalakhu/BPMF.html
"""

import time
import numpy as np
import numpy.random as rand
from numpy.linalg import inv, cholesky

from base import Base, DimensionError
from ..util.load_data import load_ml_1m, load_rating_matrix
from ..util.distributions import wishartrand
from ..util.evaluation_metrics import RMSE


class BayesianMatrixFactorization(Base):

    def __init__(self, num_user, num_item, num_feature, train, validation, **params):
        super(BayesianMatrixFactorization, self).__init__()

        self.num_user = num_user
        self.num_item = num_item
        self.num_feature = num_feature
        self.train = train
        self.validation = validation

        self.mean_rating = np.mean(self.train[:, 2])

        self.max_rating = params.get('max_rating')
        self.min_rating = params.get('min_rating')
        if self.max_rating:
            self.max_rating = float(self.max_rating)
        if self.min_rating:
            self.min_rating = float(self.min_rating)

        # Hyper Parameter
        self.beta = float(params.get('beta', 2.0))
        # Inv-Whishart (User features)
        self.WI_user = np.eye(num_feature, dtype='float16')
        self.beta_user = float(params.get('beta_user', 2.0))
        self.df_user = int(params.get('df_user', num_feature))
        self.mu_user = np.zeros((num_feature, 1), dtype='float16')

        # Inv-Whishart (item features)
        self.WI_item = np.eye(num_feature, dtype='float16')
        self.beta_item = float(params.get('beta_item', 2.0))
        self.df_item = int(params.get('df_item', num_feature))
        self.mu_item = np.zeros((num_feature, 1), dtype='float16')

        # Latent Variables
        self.mu_user = np.zeros((num_feature, 1), dtype='float16')
        self.mu_item = np.zeros((num_feature, 1), dtype='float16')

        self.alpha_user = np.eye(num_feature, dtype='float16')
        self.alpha_item = np.eye(num_feature, dtype='float16')

        self.user_features = 0.3 * np.random.rand(num_user, num_feature)
        self.item_features = 0.3 * np.random.rand(num_item, num_feature)

        self.matrix = load_rating_matrix()

    def estimate(self, iterations=100, tolerance=1e-5):
        last_rmse = None

        # the algorithm will converge, but really slow
        # use MF's initialize latent parameter will be better
        for iteration in xrange(iterations):
            # update item & user parameter
            self._update_item_params()
            self._update_user_params()

            # update item & user_features
            self._udpate_item_features()
            self._update_user_features()

            # compute RMSE
            # train errors
            train_preds = self.predict(self.train)
            train_rmse = RMSE(train_preds, np.float16(self.train[:, 2]))

            # validation errors
            validation_preds = self.predict(self.validation)
            validation_rmse = RMSE(
                validation_preds, np.float16(self.validation[:, 2]))
            self.train_errors.append(train_rmse)
            self.validation_erros.append(validation_rmse)
            print "iterations: %3d, train RMSE: %.6f, validation RMSE: %.6f " % (iteration + 1, train_rmse, validation_rmse)

            # stop if converge
            if last_rmse:
                if abs(train_rmse - last_rmse) < tolerance:
                    break

            last_rmse = train_rmse

    def predict(self, data):
        u_features = self.user_features[data[:, 0], :]
        i_features = self.item_features[data[:, 1], :]
        preds = np.sum(u_features * i_features, 1) + self.mean_rating

        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating

        return preds

    def _update_item_params(self):
        N = self.num_item
        X_bar = np.mean(self.item_features, 0)
        X_bar = np.reshape(X_bar, (self.num_feature, 1))
        # print 'X_bar', X_bar.shape
        S_bar = np.cov(self.item_features.T)
        # print 'S_bar', S_bar.shape

        norm_X_bar = X_bar - self.mu_item
        # print 'norm_X_bar', norm_X_bar.shape

        WI_post = self.WI_item + N * S_bar + \
            np.dot(norm_X_bar, norm_X_bar.T) * \
            (N * self.beta_item) / (self.beta_item + N)
        # print 'WI_post', WI_post.shape

        # Not sure why we need this...
        WI_post = (WI_post + WI_post.T) / 2.0
        df_post = self.df_item + N

        # update alpha_item
        self.alpha_item = wishartrand(df_post, WI_post)

        # update mu_item
        mu_temp = (self.beta_item * self.mu_item + N * X_bar) / \
            (self.beta_item + N)
        # print "mu_temp", mu_temp.shape
        lam = cholesky(inv(np.dot(self.beta_item + N, self.alpha_item)))
        # print 'lam', lam.shape
        self.mu_item = mu_temp + np.dot(lam, rand.randn(self.num_feature, 1))
        # print 'mu_item', self.mu_item.shape

    def _update_user_params(self):
        # same as _update_user_params
        N = self.num_user
        X_bar = np.mean(self.user_features, 0).T
        X_bar = np.reshape(X_bar, (self.num_feature, 1))

        # print 'X_bar', X_bar.shape
        S_bar = np.cov(self.user_features.T)
        # print 'S_bar', S_bar.shape

        norm_X_bar = X_bar - self.mu_user
        # print 'norm_X_bar', norm_X_bar.shape

        WI_post = self.WI_user + N * S_bar + \
            np.dot(norm_X_bar, norm_X_bar.T) * \
            (N * self.beta_user) / (self.beta_user + N)
        # print 'WI_post', WI_post.shape

        # Not sure why we need this...
        WI_post = (WI_post + WI_post.T) / 2.0
        df_post = self.df_user + N

        # update alpha_user
        self.alpha_user = wishartrand(df_post, WI_post)

        # update mu_item
        mu_temp = (self.beta_user * self.mu_user + N * X_bar) / \
            (self.beta_user + N)
        # print 'mu_temp', mu_temp.shape
        lam = cholesky(inv(np.dot(self.beta_user + N, self.alpha_user)))
        # print 'lam', lam.shape
        self.mu_user = mu_temp + np.dot(lam, rand.randn(self.num_feature, 1))
        # print 'mu_user', self.mu_user.shape

    def _udpate_item_features(self):
        # Gibbs sampling for item features
        for item_id in xrange(self.num_item):
            vec = self.matrix[:, item_id] > 0.0
            # print 'vec', vec.shape
            # if vec.shape[0] == 0:
            #    continue
            features = self.user_features[vec, :]
            # print 'features', features.shape
            rating = self.matrix[vec, item_id] - self.mean_rating
            rating_len = len(rating)
            rating = np.reshape(rating, (rating_len, 1))

            # print 'rating', rating.shape
            covar = inv(
                self.alpha_item + self.beta * np.dot(features.T, features))
            # print 'covar', covar.shape
            lam = cholesky(covar)

            temp = self.beta * \
                np.dot(features.T, rating) + np.dot(
                    self.alpha_item, self.mu_item)
            # print 'temp', temp.shape
            mean = np.dot(covar, temp)
            # print 'mean', mean.shape
            temp_feature = mean + np.dot(lam, rand.randn(self.num_feature, 1))
            temp_feature = np.reshape(temp_feature, (self.num_feature,))
            self.user_features[item_id, :] = temp_feature

    def _update_user_features(self):
        self.matrix = self.matrix.T
        # Gibbs sampling for user features
        for user_id in xrange(self.num_user):
            vec = self.matrix[:, user_id] > 0.0
            # print len(vec)
            # if vec.shape[0] == 0:
            #    continue
            # print "item_feature", self.item_features.shape
            features = self.item_features[vec, :]
            rating = self.matrix[vec, user_id] - self.mean_rating
            rating_len = len(rating)
            rating = np.reshape(rating, (rating_len, 1))

            # print 'rating', rating.shape
            covar = inv(
                self.alpha_user + self.beta * np.dot(features.T, features))
            lam = cholesky(covar)
            temp = self.beta * \
                np.dot(features.T, rating) + np.dot(
                    self.alpha_user, self.mu_user)
            mean = np.dot(covar, temp)
            # print 'mean', mean.shape
            temp_feature = mean + np.dot(lam, rand.randn(self.num_feature, 1))
            temp_feature = np.reshape(temp_feature, (self.num_feature,))
            self.user_features[user_id, :] = temp_feature

        # transpose back
        self.matrix = self.matrix.T

    def suggestions(self, user_id, num=10):
        # TODO
        pass

    def save_model(self):
        # TODO
        pass

    def load_model(self):
        # TODO
        pass

    def load_features(self, path):
        import cPickle
        import gzip
        with gzip.open(path, 'rb') as f:
            self._user_features = cPickle.load(f)
            self._item_features = cPickle.load(f)
            num_user, num_feature_u = self._user_features.shape
            num_item, num_feature_i = self._item_features.shape

            if num_feature_i != num_feature_u:
                raise DimensionError()
            self._num_feature = num_feature_i

        return self

    def save_features(self, path):
        import cPickle
        import gzip
        with gzip.open(path, 'wb') as f:
            cPickle.dump(
                self._user_features, f, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(
                self._item_features, f, protocol=cPickle.HIGHEST_PROTOCOL)


def example():
    """simple test and performance measure
    """
    num_user, num_item, ratings = load_ml_1m()
    # suffle_data
    np.random.shuffle(ratings)

    # split data to training & validation
    train_pct = 0.9
    train_size = int(train_pct * len(ratings))
    train = ratings[:train_size]
    validation = ratings[train_size:]

    # params
    num_feature = 10
    bmf_model = BayesianMatrixFactorization(
        num_user, num_item, num_feature, train, validation, max_rating=5, min_rating=1)

    start_time = time.clock()
    bmf_model.estimate(5)
    end_time = time.clock()
    print "time spend = %.3f" % (end_time - start_time)

    return bmf_model
