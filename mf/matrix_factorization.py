"""
Reference paper: "Probabilistic Matrix Factorization"
                 R. Salakhutdinov and A.Mnih.  
                 Neural Information Processing Systems 21 (NIPS 2008). Jan. 2008.

Reference Matlab code: http://www.cs.toronto.edu/~rsalakhu/BPMF.html
"""

import numpy as np

from base import Base
from util.load_data import load_ml_1m
from util.evaluation_metrics import RMSE

class MatrixFactorization(Base):

    def __init__(self, num_user, num_item, num_feature, train, validation, **params):
        super(MatrixFactorization, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.num_feature = num_feature
        self.train = train
        self.validation = validation

        # batch size
        self.batch_size = int(params.get('batch_size',100000))
        
        # learning rate
        self.epsilon = float(params.get('epsilon', 50.0))
        # regularization parameter (lambda)
        self.lam = float(params.get('lam', 0.001))

        self.max_rating = params.get('max_rating')
        self.min_rating = params.get('min_rating')
        if self.max_rating:
            self.max_rating = float(self.max_rating)
        if self.min_rating:
            self.min_rating = float(self.min_rating)

        # mean rating
        self.mean_ratings = np.mean(self.train[:, 2])

        # latent variables
        self.user_features = 0.3 * np.random.rand(num_user, num_feature)
        self.item_features = 0.3 * np.random.rand(num_item, num_feature)

    def estimate(self, iterations=100, tolerance=1e-6):
        last_rmse = None
        batch_num = int(np.ceil(float(len(self.train) / self.batch_size)))
        print "batch_num =", batch_num

        for iteration in xrange(iterations):
            np.random.shuffle(self.train)

            for batch in xrange(batch_num):
                start_index = self.batch_size * batch
                end_index = min(start_index + self.batch_size, len(self.train))
                data = self.train[start_index:end_index]
                #print "data", data.shape
                

                # compute gradient
                u_features = self.user_features[data[:, 0], :]
                i_features = self.item_features[data[:, 1], :]
                #print "u_feature", u_features.shape
                #print "i_feature", i_features.shape
                ratings = data[:, 2] - self.mean_ratings
                preds = np.sum(u_features * i_features, 1) 
                errs = preds - ratings
                err_mat = np.tile(errs, (self.num_feature, 1)).T
                #print "err_mat", err_mat.shape

                u_grads = u_features * err_mat + self.lam * i_features
                i_grads = i_features * err_mat + self.lam * u_features

                u_feature_grads = np.zeros((self.num_user, self.num_feature))
                i_feature_grads = np.zeros((self.num_item, self.num_feature))

                for i in xrange(self.batch_size):
                    user = data[i, 0]
                    item = data[i, 1]
                    u_feature_grads[user, :] += u_grads[i, :]
                    i_feature_grads[item, :] += i_grads[i, :]

                # update latent variables
                self.user_features = self.user_features - (self.epsilon / self.batch_size) * u_feature_grads
                self.item_features = self.item_features - (self.epsilon / self.batch_size) * i_feature_grads

            # compute RMSE
            # train errors
            train_preds = self.predict(self.train)
            train_rmse = RMSE(train_preds, np.float16(self.train[:, 2]))

            # validation errors
            validation_preds = self.predict(self.validation)
            validation_rmse = RMSE(validation_preds, np.float16(self.validation[:, 2]))
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
        preds = np.sum(u_features * i_features, 1) + self.mean_ratings

        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating

        return preds

    def suggestions(self, user_id, num=10):
        pass

def test():
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
    
    mf_model = MatrixFactorization(num_user, num_item, num_feature, train, validation, max_rating=5, min_rating=1)
    mf_model.estimate()

    return mf_model


