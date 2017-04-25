import os
import unittest
import numpy as np
import numpy.testing as np_test
import scipy.sparse as sparse

from six.moves import xrange

from recommend.utils.datasets import (load_movielens_1m_ratings,
                                      load_movielens_100k_ratings,
                                      make_ratings,
                                      build_user_item_matrix)

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
TEST_ML_1M_RATING_FILE = "ml_1m_ratings_sample_1k.dat"
TEST_ML_100K_RATING_FILE = "ml_100k_ratings_sample_1k.dat"


class TestLoadData(unittest.TestCase):

    def test_load_movielens_1m_ratings(self):
        test_rating_file = os.path.join(TEST_DATA_DIR, TEST_ML_1M_RATING_FILE)
        ratings = load_movielens_1m_ratings(test_rating_file)

        n_row, n_col = ratings.shape
        self.assertEqual(n_row, 1000)
        self.assertEqual(n_col, 3)
        np_test.assert_array_equal(ratings[0], [1, 1193, 5])
        np_test.assert_array_equal(ratings[-1], [10, 1022, 5])

    def test_load_movielens_100k_ratings(self):
        test_rating_file = os.path.join(
            TEST_DATA_DIR, TEST_ML_100K_RATING_FILE)
        ratings = load_movielens_100k_ratings(test_rating_file)

        n_row, n_col = ratings.shape
        self.assertEqual(n_row, 1000)
        self.assertEqual(n_col, 3)
        np_test.assert_array_equal(ratings[0], [196, 242, 3])
        np_test.assert_array_equal(ratings[-1], [59, 485, 2])


class TestMakeData(unittest.TestCase):

    def test_make_ratings(self):
        user_size = [10, 20, 50]
        item_size = [50, 100, 200]
        min_cnts = [1, 5, 10]
        max_cnts = [5, 10, 15]
        choices = list(range(1, 10))
        params = zip(user_size, item_size, min_cnts, max_cnts)
        for (n_user, n_item, min_cnt, max_cnt) in params:
            ratings = make_ratings(n_user, n_item, min_cnt, max_cnt, choices)
            self.assertTrue(isinstance(ratings, np.ndarray))
            self.assertTrue(int(ratings[:, 0].max()) < n_user)
            self.assertTrue(int(ratings[:, 1].max()) < n_item)
            self.assertTrue(ratings[:, 2].max() <= max(choices))
            self.assertTrue(ratings[:, 2].min() >= min(choices))

    def test_make_ratings_input_check(self):
        with self.assertRaises(ValueError):
            make_ratings(10, 10, 5, 10, [1, 2, 3])

        with self.assertRaises(ValueError):
            make_ratings(10, 10, 5, 4, [1, 2, 3])

        with self.assertRaises(ValueError):
            make_ratings(10, 10, 5, 6, 2)

    def test_build_user_item_matrix(self):
        n_user = 200
        n_item = 300
        choices = list(range(1, 5))
        ratings = make_ratings(n_user, n_item, 5, 10, choices)
        mtx = build_user_item_matrix(n_user, n_item, ratings)
        self.assertTrue(sparse.issparse(mtx))
        self.assertEqual(mtx.shape[0], n_user)
        self.assertEqual(mtx.shape[1], n_item)
        dense_mtx = mtx.toarray()
        for i in xrange(ratings.shape[0]):
            user_idx = ratings[i][0]
            item_idx = ratings[i][1]
            rating = ratings[i][2]
            np_test.assert_almost_equal(dense_mtx[user_idx, item_idx], rating)
