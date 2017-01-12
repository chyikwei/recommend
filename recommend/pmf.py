"""
Reference paper:
    "Probabilistic Matrix Factorization"
    R. Salakhutdinov and A.Mnih.
    Neural Information Processing Systems 21 (NIPS 2008). Jan. 2008.

Reference Matlab code: http://www.cs.toronto.edu/~rsalakhu/BPMF.html
"""

import logging

import numpy as np
from numpy.random import RandomState
from .base import ModelBase
from .exceptions import NotFittedError
from .utils.evaluation import RMSE

logger = logging.getLogger(__name__)


class PMF(ModelBase):
    """Probabilistic Matrix Factorization
    """

    def __init__(self, n_user, n_item, n_feature, batch_size=1e5, epsilon=100.0,
                 seed=None, reg=1e-2, converge=1e-5, max_rating=None, min_rating=None):

        super(PMF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_feature = n_feature

        self.random_state = RandomState(seed)

        # batch size
        self.batch_size = batch_size

        # learning rate
        self.epsilon = epsilon
        # regularization parameter
        self.reg = reg
        self.converge = converge
        self.max_rating = float(max_rating) if max_rating is not None else max_rating
        self.min_rating = float(min_rating) if min_rating is not None else min_rating

        # data state
        self._mean_rating = None
        # user/item features
        self._user_features = 0.3 * self.random_state.rand(n_user, n_feature)
        self._item_features = 0.3 * self.random_state.rand(n_item, n_feature)

    def fit(self, train, n_iters=50):

        self._mean_rating = np.mean(train[:, 2])
        last_rmse = None
        batch_num = int(np.ceil(float(len(train) / self.batch_size)))
        logger.info("batch count = %d", batch_num + 1)

        u_feature_grads = np.zeros((self.n_user, self.n_feature))
        i_feature_grads = np.zeros((self.n_item, self.n_feature))
        for iteration in xrange(n_iters):
            #self.random_state.shuffle(train)
            logger.info("iteration %d...", iteration)

            for batch in xrange(batch_num):
                start_idx = int(batch * self.batch_size)
                end_idx = int((batch + 1) * self.batch_size)
                data = train[start_idx:end_idx]
                # print "data", data.shape

                # compute gradient
                u_features = self._user_features[data[:, 0], :]
                i_features = self._item_features[data[:, 1], :]
                # print "u_feature", u_features.shape
                # print "i_feature", i_features.shape
                ratings = data[:, 2] - self._mean_rating

                preds = np.sum(u_features * i_features, 1)
                errs = preds - ratings
                err_mat = np.tile(errs, (self.n_feature, 1)).T
                # print "err_mat", err_mat.shape

                u_grads = u_features * err_mat + self.reg * u_features
                i_grads = i_features * err_mat + self.reg * i_features

                u_feature_grads.fill(0.0)
                i_feature_grads.fill(0.0)
                for i in xrange(data.shape[0]):
                    user = data[i, 0]
                    item = data[i, 1]
                    u_feature_grads[user, :] += u_grads[i, :]
                    i_feature_grads[item, :] += i_grads[i, :]

                # update latent variables
                self._user_features = self._user_features - \
                    (self.epsilon / self.batch_size) * u_feature_grads
                self._item_features = self._item_features - \
                    (self.epsilon / self.batch_size) * i_feature_grads

            # compute RMSE
            train_preds = self.predict(train[:, :2])
            train_rmse = RMSE(train_preds, train[:, 2])
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

        u_features = self._user_features[data[:, 0], :]
        i_features = self._item_features[data[:, 1], :]
        preds = np.sum(u_features * i_features, 1) + self._mean_rating

        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating
        return preds
