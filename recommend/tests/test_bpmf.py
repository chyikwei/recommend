import os
import unittest

from ..utils.datasets import make_ratings
from ..utils.evaluation import RMSE
from ..bpmf import BPMF
from ..exceptions import NotFittedError


class TestBPMF(unittest.TestCase):

    def setUp(self):
        self.n_feature = 10
        self.rating_choices = list(range(1, 5))
        self.max_rat = max(self.rating_choices)
        self.min_rat = min(self.rating_choices)
        self.seed = 0

    def test_bpmf_with_random_data(self):
        n_user = 1000
        n_item = 2000
        ratings = make_ratings(n_user, n_item, 20, 30, self.rating_choices, seed=self.seed)

        bpmf1 = BPMF(n_user, n_item, self.n_feature,
                     max_rating=self.max_rat,
                     min_rating=self.min_rat,
                     seed=self.seed)

        bpmf1.fit(ratings, n_iters=1)
        rmse_1 = RMSE(bpmf1.predict(ratings[:, :2]), ratings[:, 2])

        bpmf2 = BPMF(n_user, n_item, self.n_feature,
                     max_rating=self.max_rat,
                     min_rating=self.min_rat,
                     seed=self.seed)

        bpmf2.fit(ratings, n_iters=10)
        rmse_2 = RMSE(bpmf2.predict(ratings[:, :2]), ratings[:, 2])
        self.assertTrue(rmse_1 > rmse_2)

    def test_not_fitted_err(self):
        with self.assertRaises(NotFittedError):
            ratings = make_ratings(10, 10, 1, 5, self.rating_choices, seed=self.seed)
            bpmf = BPMF(10, 10, self.n_feature)
            bpmf.predict(ratings[:, :2])
