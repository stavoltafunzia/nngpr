import unittest

from nngpr.cunngpr import CUNNGPR
from .nngpr_tester import NNGPRTester


class TestCUNNGPR(NNGPRTester, unittest.TestCase):

    def get_gpr(self, fixed_bounds=False, kernel=None, num_nn=8, **kwargs):
        """Helper function that returns the NNGPR instance"""
        if kernel is None:
            kernel = self.get_kern(fixed_bounds=fixed_bounds)
        gpr = CUNNGPR(kernel=kernel, num_nn=num_nn, batch_size=7, **kwargs)
        return gpr


if __name__ == '__main__':
    unittest.main()
