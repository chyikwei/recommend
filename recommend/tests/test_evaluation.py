import unittest

import numpy as np
import numpy.testing as np_test
from numpy.random import RandomState


from recommend.utils.evaluation import RMSE


class TestRMSE(unittest.TestCase):

    def test_rmse_same_input(self):
        rs = RandomState(0)
        data = rs.randn(100)
        np_test.assert_almost_equal(RMSE(data, data), 0.)

    def test_rmse(self):
        np_test.assert_almost_equal(
            RMSE(np.ones(100), np.zeros(100)), np.sqrt(100. / 99.))
