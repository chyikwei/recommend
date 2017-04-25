"""
Reference: "Large-scale Parallel Collaborative Filtering for the Netflix Prize"
            Y. Zhou, D. Wilkinson, R. Schreiber and R. Pan, 2008
"""

import logging
from six.moves import xrange
import numpy as np
from numpy.random import RandomState
from numpy.linalg import inv

from .base import ModelBase
from .exceptions import NotFittedError
from .utils.datasets import build_user_item_matrix
from .utils.validation import check_ratings
from .utils.evaluation import RMSE

logger = logging.getLogger(__name__)


class ALS(ModelBase):
    """Alternating Least Squares with Weighted Lambda Regularization (ALS-WR)
    """

    def __init__(self, n_user, n_item, n_feature, reg=1e-2, converge=1e-5,
                 seed=None, max_rating=None, min_rating=None):
        super(ALS, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_feature = n_feature
        self.reg = float(reg)
        self.rand_state = RandomState(seed)
        self.max_rating = float(max_rating) if max_rating is not None else None
        self.min_rating = float(min_rating) if min_rating is not None else None
        self.converge = converge

        # data state
        self.mean_rating_ = None
        self.ratings_csr_ = None
        self.ratings_csc_ = None

        # user/item features
        self.user_features_ = 0.1 * self.rand_state.rand(n_user, n_feature)
        self.item_features_ = 0.1 * self.rand_state.rand(n_item, n_feature)

    def _update_user_feature(self):
        """Fix item features and update user features
        """
        for i in xrange(self.n_user):
            _, item_idx = self.ratings_csr_[i, :].nonzero()
            # number of ratings of user i
            n_u = item_idx.shape[0]
            if n_u == 0:
                logger.debug("no ratings for user %d", i)
                continue
            item_features = self.item_features_.take(item_idx, axis=0)
            ratings = self.ratings_csr_[i, :].data - self.mean_rating_

            A_i = (np.dot(item_features.T, item_features) +
                   self.reg * n_u * np.eye(self.n_feature))
            V_i = np.dot(item_features.T, ratings)
            self.user_features_[i, :] = np.dot(inv(A_i), V_i)

    def _update_item_feature(self):
        """Fix user features and update item features
        """
        for j in xrange(self.n_item):
            user_idx, _ = self.ratings_csc_[:, j].nonzero()
            # number of ratings of item j
            n_i = user_idx.shape[0]
            if n_i == 0:
                logger.debug("no ratings for item %d", j)
                continue
            user_features = self.user_features_.take(user_idx, axis=0)
            ratings = self.ratings_csc_[:, j].data - self.mean_rating_

            A_j = (np.dot(user_features.T, user_features) +
                   self.reg * n_i * np.eye(self.n_feature))
            V_j = np.dot(user_features.T, ratings)
            self.item_features_[j, :] = np.dot(inv(A_j), V_j)

    def fit(self, ratings, n_iters=50):

        check_ratings(ratings, self.n_user, self.n_item,
                      self.max_rating, self.min_rating)
        self.mean_rating_ = np.mean(ratings.take(2, axis=1))
        # csr user-item matrix for fast row access (user update)
        self.ratings_csr_ = build_user_item_matrix(
            self.n_user, self.n_item, ratings)
        # keep a csc matrix for fast col access (item update)
        self.ratings_csc_ = self.ratings_csr_.tocsc()

        last_rmse = None
        for iteration in xrange(n_iters):
            logger.debug("iteration %d...", iteration)

            self._update_user_feature()
            self._update_item_feature()

            # compute RMSE
            train_preds = self.predict(ratings.take([0, 1], axis=1))
            train_rmse = RMSE(train_preds, ratings.take(2, axis=1))
            logger.info("iter: %d, train RMSE: %.6f", iteration, train_rmse)

            # stop when converge
            if last_rmse and abs(train_rmse - last_rmse) < self.converge:
                logger.info('converges at iteration %d. stop.', iteration)
                break
            else:
                last_rmse = train_rmse

    def predict(self, data):

        if not self.mean_rating_:
            raise NotFittedError("Please fit model before run predict")

        u_features = self.user_features_.take(data.take(0, axis=1), axis=0)
        i_features = self.item_features_.take(data.take(1, axis=1), axis=0)
        preds = np.sum(u_features * i_features, 1) + self.mean_rating_

        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating
        return preds
