import os
import sys
import gzip
import unittest

from recommend.utils.datasets import make_ratings
from recommend.utils.evaluation import RMSE
from recommend.pmf import PMF
from recommend.exceptions import NotFittedError

if sys.version_info[0] == 3:
    import _pickle as cPickle
else:
    import cPickle

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
ML_100K_RATING_PKL = "ml_100k_ratings.pkl.gz"


class TestPMF(unittest.TestCase):

    def setUp(self):
        self.n_feature = 10
        self.rating_choices = list(range(1, 5))
        self.max_rat = max(self.rating_choices)
        self.min_rat = min(self.rating_choices)
        self.seed = 0

    def test_pmf_with_random_data(self):
        n_user = 1000
        n_item = 2000
        n_feature = self.n_feature
        ratings = make_ratings(
            n_user, n_item, 20, 30, self.rating_choices, seed=self.seed)

        pmf1 = PMF(n_user, n_item, n_feature,
                   batch_size=1000.,
                   epsilon=10.,
                   seed=0,
                   max_rating=self.max_rat,
                   min_rating=self.min_rat)

        pmf1.fit(ratings, n_iters=1)
        rmse_1 = RMSE(pmf1.predict(ratings[:, :2]), ratings[:, 2])

        pmf2 = PMF(n_user, n_item, n_feature,
                   batch_size=1000.,
                   epsilon=10.,
                   seed=0,
                   max_rating=self.max_rat,
                   min_rating=self.min_rat)

        pmf2.fit(ratings, n_iters=3)
        rmse_2 = RMSE(pmf2.predict(ratings[:, :2]), ratings[:, 2])
        self.assertTrue(rmse_1 > rmse_2)

    def test_pmf_convergence(self):
        n_user = 100
        n_item = 200
        n_feature = self.n_feature
        ratings = make_ratings(
            n_user, n_item, 20, 30, self.rating_choices, seed=self.seed)

        pmf1 = PMF(n_user, n_item, n_feature,
                   seed=0,
                   max_rating=self.max_rat,
                   min_rating=self.min_rat,
                   converge=1e-5)

        pmf1.fit(ratings, n_iters=5)
        rmse_1 = RMSE(pmf1.predict(ratings[:, :2]), ratings[:, 2])

        pmf2 = PMF(n_user, n_item, n_feature,
                   seed=0,
                   max_rating=self.max_rat,
                   min_rating=self.min_rat,
                   converge=1e-1)

        pmf2.fit(ratings, n_iters=5)
        rmse_2 = RMSE(pmf2.predict(ratings[:, :2]), ratings[:, 2])
        self.assertTrue(rmse_1 < rmse_2)

    def test_pmf_not_fitted_err(self):
        with self.assertRaises(NotFittedError):
            ratings = make_ratings(
                10, 10, 1, 5, self.rating_choices, seed=self.seed)
            bpmf = PMF(10, 10, self.n_feature)
            bpmf.predict(ratings[:, :2])


class TestPMFwithMovieLens100K(unittest.TestCase):

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

    def test_pmf_with_ml_100k_rating(self):
        n_user = 943
        n_item = 1682
        n_feature = 10
        ratings = self.ratings

        pmf = PMF(n_user, n_item, n_feature,
                  batch_size=1e4,
                  epsilon=20.,
                  reg=1e-4,
                  max_rating=5.,
                  min_rating=1.,
                  seed=self.seed)

        pmf.fit(ratings, n_iters=15)
        rmse = RMSE(pmf.predict(ratings[:, :2]), ratings[:, 2])
        self.assertTrue(rmse < 0.85)
