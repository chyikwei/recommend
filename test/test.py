import unittest
import numpy as np

from ..util.load_data import load_ml_1m, load_rating_matrix
from ..mf.matrix_factorization import MatrixFactorization as MF
from ..mf.bayesian_matrix_factorization import BayesianMatrixFactorization as BMF
from ..mf.theano.matrix_factorization import MatrixFactorization as MF_theano


class TestMF(unittest.TestCase):
    # TODO: finish all test case

    def setUp(self):
        num_user, num_item, ratings = load_ml_1m()
        np.random.shuffle(ratings)
        self.num_user = num_user
        self.num_item = num_item
        self.ratings = ratings
        self.num_feature = 10

        train_pct = 0.9
        train_size = int(train_pct * len(self.ratings))
        self.train = self.ratings[:train_size]
        self.validation = self.ratings[train_size:]

    def test_load_data(self):
        self.assertEqual(self.num_user, 6040)
        self.assertEqual(self.num_item, 3952)
        self.assertEqual(len(self.ratings), 1000209)

    def test_load_rating_matrix(self):
        matrix = load_rating_matrix()
        self.assertEqual(matrix.shape, (6040, 3952))

    def test_build_rating_matrix(self):
        pass

    def test_build_sparse_rating_matrix(self):
        pass

    def test_mf(self):
        model = MF(
            self.num_user, self.num_item, self.num_feature, self.train, self.validation, max_rating=5, min_rating=1)
        model.estimate(5)
        # training error should go down
        self.assertGreater(model.train_errors[0], model.train_errors[-1])

    def test_bayes_mf(self):
        model = BMF(
            self.num_user, self.num_item, self.num_feature, self.train, self.validation, max_rating=5, min_rating=1)
        model.estimate(5)
        # training error should go down
        self.assertGreater(model.train_errors[0], model.train_errors[-1])

    def test_mf_theano(self):
        model = MF_theano(
            self.num_user, self.num_item, self.num_feature, self.train, self.validation, max_rating=5, min_rating=1)
        model.estimate(5)
        self.assertGreater(model.train_errors[0], model.train_errors[-1])

if __name__ == '__main__':
    unittest.main()
