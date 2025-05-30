import unittest

from nngpr import batched_kernels
from nngpr.numpy_nngpr import NumpyWorker, NumpyNngpr
import numpy as np

from test.nngpr_base_tester import NngprTester, WorkerTester


class TestNumpyWorker(WorkerTester, unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(NumpyWorker(), *args, **kwargs)


class TestNumpyNngpr(NngprTester, unittest.TestCase):

    def get_gpr(self, fixed_bounds=False, kernel=None, num_nn=8, **kwargs):
        """Helper function that returns the NNGPR instance"""
        if kernel is None:
            kernel = self.get_kern(fixed_bounds=fixed_bounds)
        kwargs['batch_size'] = kwargs.get('batch_size', 7)
        gpr = NumpyNngpr(kernel=kernel, num_nn=num_nn, **kwargs)
        return gpr


if __name__ == '__main__':
    unittest.main()
