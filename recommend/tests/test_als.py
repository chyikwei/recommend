import os
import sys
import gzip
import unittest

from numpy.testing import (assert_array_equal,
                           assert_raises)
from recommend.utils.datasets import make_ratings
from recommend.utils.evaluation import RMSE
from recommend.als import ALS
from recommend.exceptions import NotFittedError

if sys.version_info[0] == 3:
    import _pickle as cPickle
else:
    import cPickle

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
ML_100K_RATING_PKL = "ml_100k_ratings.pkl.gz"


class TestALS(unittest.TestCase):

    def setUp(self):
        self.n_feature = 10
        self.rating_choices = list(range(1, 5))
        self.max_rat = max(self.rating_choices)
        self.min_rat = min(self.rating_choices)
        self.seed = 0

    def test_als_with_random_data(self):
        n_user = 100
        n_item = 200
        n_feature = self.n_feature
        ratings = make_ratings(
            n_user, n_item, 20, 30, self.rating_choices, seed=self.seed)

        als1 = ALS(n_user, n_item, n_feature,
                   reg=1e-2,
                   seed=0,
                   max_rating=self.max_rat,
                   min_rating=self.min_rat)

        als1.fit(ratings, n_iters=1)
        rmse_1 = RMSE(als1.predict(ratings[:, :2]), ratings[:, 2])

        als2 = ALS(n_user, n_item, n_feature,
                   reg=1e-2,
                   seed=0,
                   max_rating=self.max_rat,
                   min_rating=self.min_rat)

        als2.fit(ratings, n_iters=3)
        rmse_2 = RMSE(als2.predict(ratings[:, :2]), ratings[:, 2])
        self.assertTrue(rmse_1 > rmse_2)

    def test_als_convergence(self):
        n_user = 100
        n_item = 200
        n_feature = self.n_feature
        ratings = make_ratings(
            n_user, n_item, 20, 30, self.rating_choices, seed=self.seed)

        als1 = ALS(n_user, n_item, n_feature,
                   reg=1e-2,
                   seed=0,
                   max_rating=self.max_rat,
                   min_rating=self.min_rat,
                   converge=1e-2)

        als1.fit(ratings, n_iters=10)
        rmse_1 = RMSE(als1.predict(ratings[:, :2]), ratings[:, 2])

        als2 = ALS(n_user, n_item, n_feature,
                   reg=1e-2,
                   seed=0,
                   max_rating=self.max_rat,
                   min_rating=self.min_rat,
                   converge=1e-1)

        als2.fit(ratings, n_iters=10)
        rmse_2 = RMSE(als2.predict(ratings[:, :2]), ratings[:, 2])
        self.assertTrue(rmse_1 < rmse_2)

    def test_als_seed(self):
        n_user = 100
        n_item = 200
        n_feature = self.n_feature
        ratings = make_ratings(
            n_user, n_item, 20, 30, self.rating_choices, seed=self.seed)

        # seed 0
        als1 = ALS(n_user, n_item, n_feature,
                   reg=1e-2,
                   seed=0,
                   max_rating=self.max_rat,
                   min_rating=self.min_rat)
        als1.fit(ratings, n_iters=3)

        als2 = ALS(n_user, n_item, n_feature,
                   reg=1e-2,
                   seed=0,
                   max_rating=self.max_rat,
                   min_rating=self.min_rat)
        als2.fit(ratings, n_iters=3)
        assert_array_equal(als1.user_features_, als2.user_features_)
        assert_array_equal(als1.item_features_, als2.item_features_)

        # seed 1
        als3 = ALS(n_user, n_item, n_feature,
                   reg=1e-2,
                   seed=1,
                   max_rating=self.max_rat,
                   min_rating=self.min_rat)
        als3.fit(ratings, n_iters=3)
        assert_raises(AssertionError, assert_array_equal,
                      als1.user_features_, als3.user_features_)
        assert_raises(AssertionError, assert_array_equal,
                      als1.item_features_, als3.item_features_)

    def test_als_with_missing_data(self):
        n_user = 10
        n_item = 20
        n_feature = self.n_feature
        ratings = make_ratings(
            n_user - 1, n_item - 1, 5, 10, self.rating_choices, seed=self.seed)
        als1 = ALS(n_user, n_item, n_feature,
                   reg=1e-2,
                   seed=0,
                   max_rating=self.max_rat,
                   min_rating=self.min_rat)

        unuse_user_f_before = als1.user_features_[n_user - 1, :]
        unuse_item_f_before = als1.item_features_[n_item - 1, :]
        als1.fit(ratings, n_iters=1)
        unuse_user_f_after = als1.user_features_[n_user - 1, :]
        unuse_item_f_after = als1.item_features_[n_item - 1, :]
        # last user/item feature should be
        #  unchanged since no rating data on them
        assert_array_equal(unuse_user_f_before, unuse_user_f_after)
        assert_array_equal(unuse_item_f_before, unuse_item_f_after)

    def test_als_not_fitted_err(self):
        with self.assertRaises(NotFittedError):
            ratings = make_ratings(
                10, 10, 1, 5, self.rating_choices, seed=self.seed)
            als = ALS(10, 10, self.n_feature)
            als.predict(ratings[:, :2])


class TestALSwithMovieLens100K(unittest.TestCase):

    def setUp(self):
        self.seed = 0

        file_path = os.path.join(TEST_DATA_DIR, ML_100K_RATING_PKL)
        with gzip.open(file_path, 'rb') as f:
            if sys.version_info[0] == 3:
                ratings = cPickle.load(f, encoding='latin1')
            else:
                ratings = cPickle.load(f)

        self.n_user = 943
        self.n_item = 1682
        self.assertEqual(ratings.shape[0], 100000)
        self.assertEqual(ratings[:, 0].min(), 1)
        self.assertEqual(ratings[:, 0].max(), self.n_user)
        self.assertEqual(ratings[:, 1].min(), 1)
        self.assertEqual(ratings[:, 1].max(), self.n_item)

        # let user_id / item_id start from 0
        ratings[:, 0] = ratings[:, 0] - 1
        ratings[:, 1] = ratings[:, 1] - 1
        self.ratings = ratings

    def test_als_with_ml_100k_rating(self):
        n_user = 943
        n_item = 1682
        n_feature = 10
        ratings = self.ratings

        als = ALS(n_user, n_item, n_feature,
                  reg=1e-2,
                  max_rating=5.,
                  min_rating=1.,
                  seed=self.seed)

        als.fit(ratings, n_iters=5)
        rmse = RMSE(als.predict(ratings[:, :2]), ratings[:, 2])
        self.assertTrue(rmse < 0.8)
