"""
Reference paper: "Bayesian Probabilistic Matrix Factorization using MCMC"
                 R. Salakhutdinov and A.Mnih.  
                 25th International Conference on Machine Learning (ICML-2008)

Reference Matlab code: http://www.cs.toronto.edu/~rsalakhu/BPMF.html
"""

import logging
from six.moves import xrange
import numpy as np
from numpy.linalg import inv, cholesky
from numpy.random import RandomState
from scipy.stats import wishart

from .base import ModelBase
from .exceptions import NotFittedError
from .utils.datasets import build_user_item_matrix
from .utils.validation import check_ratings
from .utils.evaluation import RMSE

logger = logging.getLogger(__name__)


class BPMF(ModelBase):
    """Bayesian Probabilistic Matrix Factorization
    """

    def __init__(self, n_user, n_item, n_feature, beta=2.0, beta_user=2.0,
                 df_user=None, beta_item=2.0, df_item=None, converge=1e-5,
                 seed=None, max_rating=None, min_rating=None):

        super(BPMF, self).__init__()

        self.n_user = n_user
        self.n_item = n_item
        self.n_feature = n_feature
        self.rand_state = RandomState(seed)
        self.max_rating = float(max_rating) if max_rating is not None else None
        self.min_rating = float(min_rating) if min_rating is not None else None
        self.converge = converge

        # Hyper Parameter
        self.beta = beta

        # Inv-Whishart (User features)
        self.WI_user = np.eye(n_feature, dtype='float64')
        self.beta_user = beta_user
        self.df_user = int(df_user) if df_user is not None else n_feature
        self.mu_user = np.zeros((n_feature, 1), dtype='float64')

        # Inv-Whishart (item features)
        self.WI_item = np.eye(n_feature, dtype='float64')
        self.beta_item = beta_item
        self.df_item = int(df_item) if df_item is not None else n_feature
        self.mu_item = np.zeros((n_feature, 1), dtype='float64')

        # Latent Variables
        self.mu_user = np.zeros((n_feature, 1), dtype='float64')
        self.mu_item = np.zeros((n_feature, 1), dtype='float64')

        self.alpha_user = np.eye(n_feature, dtype='float64')
        self.alpha_item = np.eye(n_feature, dtype='float64')

        self.user_features = 0.3 * self.rand_state.rand(n_user, n_feature)
        self.item_features = 0.3 * self.rand_state.rand(n_item, n_feature)

        # data state
        self._mean_rating = None
        self._ratings_csr = None
        self._ratings_csc = None

    def fit(self, ratings, n_iters=50):
        """training models"""

        check_ratings(ratings, self.n_user, self.n_item, self.max_rating, self.min_rating)

        self._mean_rating = np.mean(ratings[:, 2])

        # csr user-item matrix for fast row access (user update)
        self._ratings_csr = build_user_item_matrix(self.n_user, self.n_item, ratings)
        # keep a csc matrix for fast col access (item update)
        self._ratings_csc = self._ratings_csr.tocsc()

        last_rmse = None
        for iteration in xrange(n_iters):
            logger.info("iteration %d...", iteration)

            # update item & user parameter
            self._update_item_params()
            self._update_user_params()

            # update item & user_features
            self._udpate_item_features()
            self._update_user_features()

            # compute RMSE
            train_preds = self.predict(ratings[:, :2])
            train_rmse = RMSE(train_preds, ratings[:, 2])
            train_preds = self.predict(ratings[:, :2])
            train_rmse = RMSE(train_preds, ratings[:, 2])
            logger.info("iter: %d, train RMSE: %.6f", iteration, train_rmse)

            # stop when converge
            if last_rmse and abs(train_rmse - last_rmse) < self.converge:
                logger.info('converges at iteration %d. stop.', iteration)
                break
            else:
                last_rmse = train_rmse
        return self

    def predict(self, data):

        if not self._mean_rating:
            raise NotFittedError()

        u_features = self.user_features[data[:, 0], :]
        i_features = self.item_features[data[:, 1], :]
        preds = np.sum(u_features * i_features, 1) + self._mean_rating

        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating

        return preds

    def _update_item_params(self):
        N = self.n_item
        X_bar = np.mean(self.item_features, 0)
        X_bar = np.reshape(X_bar, (self.n_feature, 1))
        # print 'X_bar', X_bar.shape
        S_bar = np.cov(self.item_features.T)
        # print 'S_bar', S_bar.shape

        norm_X_bar = X_bar - self.mu_item
        # print 'norm_X_bar', norm_X_bar.shape

        WI_post = inv(inv(self.WI_item) + N * S_bar + \
            np.dot(norm_X_bar, norm_X_bar.T) * \
            (N * self.beta_item) / (self.beta_item + N))
        # print 'WI_post', WI_post.shape

        # Not sure why we need this...
        WI_post = (WI_post + WI_post.T) / 2.0
        df_post = self.df_item + N

        # update alpha_item
        self.alpha_item = wishart.rvs(df_post, WI_post, 1, self.rand_state)

        # update mu_item
        mu_temp = (self.beta_item * self.mu_item + N * X_bar) / \
            (self.beta_item + N)
        # print "mu_temp", mu_temp.shape
        lam = cholesky(inv(np.dot(self.beta_item + N, self.alpha_item)))
        # print 'lam', lam.shape
        self.mu_item = mu_temp + np.dot(lam, self.rand_state.randn(self.n_feature, 1))
        # print 'mu_item', self.mu_item.shape

    def _update_user_params(self):
        # same as _update_user_params
        N = self.n_user
        X_bar = np.mean(self.user_features, 0).T
        X_bar = np.reshape(X_bar, (self.n_feature, 1))

        # print 'X_bar', X_bar.shape
        S_bar = np.cov(self.user_features.T)
        # print 'S_bar', S_bar.shape

        norm_X_bar = X_bar - self.mu_user
        # print 'norm_X_bar', norm_X_bar.shape

        WI_post = inv(inv(self.WI_user) + N * S_bar + \
            np.dot(norm_X_bar, norm_X_bar.T) * \
            (N * self.beta_user) / (self.beta_user + N))
        # print 'WI_post', WI_post.shape

        # Not sure why we need this...
        WI_post = (WI_post + WI_post.T) / 2.0
        df_post = self.df_user + N

        # update alpha_user
        self.alpha_user = wishart.rvs(df_post, WI_post, 1, self.rand_state)

        # update mu_item
        mu_temp = (self.beta_user * self.mu_user + N * X_bar) / \
            (self.beta_user + N)
        # print 'mu_temp', mu_temp.shape
        lam = cholesky(inv(np.dot(self.beta_user + N, self.alpha_user)))
        # print 'lam', lam.shape
        self.mu_user = mu_temp + np.dot(lam, self.rand_state.randn(self.n_feature, 1))
        # print 'mu_user', self.mu_user.shape

    def _udpate_item_features(self):
        # Gibbs sampling for item features
        for item_id in xrange(self.n_item):
            indices = self._ratings_csc[:, item_id].indices
            # print 'vec', vec.shape
            # if vec.shape[0] == 0:
            #    continue
            features = self.user_features[indices, :]
            # print 'features', features.shape
            rating = self._ratings_csc[:, item_id].data - self._mean_rating
            rating = np.reshape(rating, (rating.shape[0], 1))

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
            temp_feature = mean + np.dot(lam, self.rand_state.randn(self.n_feature, 1))
            temp_feature = np.reshape(temp_feature, (self.n_feature,))
            self.item_features[item_id, :] = temp_feature

    def _update_user_features(self):
        # Gibbs sampling for user features
        for user_id in xrange(self.n_user):
            indices = self._ratings_csr[user_id, :].indices
            # print len(vec)
            # if vec.shape[0] == 0:
            #    continue
            # print "item_feature", self.item_features.shape
            features = self.item_features[indices, :]
            rating = self._ratings_csr[user_id, :].data - self._mean_rating
            rating = np.reshape(rating, (rating.shape[0], 1))

            # print 'rating', rating.shape
            covar = inv(
                self.alpha_user + self.beta * np.dot(features.T, features))
            lam = cholesky(covar)
            temp = self.beta * \
                np.dot(features.T, rating) + np.dot(
                    self.alpha_user, self.mu_user)
            mean = np.dot(covar, temp)
            # print 'mean', mean.shape
            temp_feature = mean + np.dot(lam, self.rand_state.randn(self.n_feature, 1))
            temp_feature = np.reshape(temp_feature, (self.n_feature,))
            self.user_features[user_id, :] = temp_feature
