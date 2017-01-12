import os
import unittest

from ..utils.datasets import make_ratings
from ..utils.evaluation import RMSE
from ..pmf import PMF
from ..exceptions import NotFittedError


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
        ratings = make_ratings(n_user, n_item, 20, 30, self.rating_choices, seed=self.seed)

        pmf1 = PMF(n_user, n_item, n_feature,
                   batch_size=1000.,
                   epsilon=100.,
                   seed=0,
                   max_rating=self.max_rat,
                   min_rating=self.min_rat)

        pmf1.fit(ratings, n_iters=1)
        rmse_1 = RMSE(pmf1.predict(ratings[:, :2]), ratings[:, 2])

        pmf2 = PMF(n_user, n_item, n_feature,
                   batch_size=1000.,
                   epsilon=100.,
                   seed=0,
                   max_rating=self.max_rat,
                   min_rating=self.min_rat)

        pmf2.fit(ratings, n_iters=10)
        rmse_2 = RMSE(pmf2.predict(ratings[:, :2]), ratings[:, 2])
        self.assertTrue(rmse_1 > rmse_2)

    def test_not_fitted_err(self):
        with self.assertRaises(NotFittedError):
            ratings = make_ratings(10, 10, 1, 5, self.rating_choices, seed=self.seed)
            bpmf = PMF(10, 10, self.n_feature)
            bpmf.predict(ratings[:, :2])
