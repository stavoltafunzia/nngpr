import unittest

try:
    import cupy as cp
    from nngpr.cupy_nngpr import CudaWorker, CupyNngpr, CupyProxy
except ImportError:
    raise unittest.SkipTest("Requires nngpr[cupy]")

from nngpr.test.nngpr_base_tester import NngprTester, WorkerTester
from nngpr.test.test_batched_kernels import TestBatchedKernels


class TestBatchedKernelsCuda(TestBatchedKernels):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, xp=CupyProxy(0), **kwargs)


class TestCudaWorker(WorkerTester, unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(CudaWorker(0, self.get_kern(), accelerate_kernel=False), *args, **kwargs)


class TestCudaWorkerAcc(WorkerTester, unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(CudaWorker(0, self.get_kern(), accelerate_kernel=True), *args, **kwargs)


class TestCupyNngpr(NngprTester, unittest.TestCase):

    def get_gpr(self, fixed_bounds=False, kernel=None, num_nn=8, **kwargs):
        """Helper function that returns the NNGPR instance"""
        if kernel is None:
            kernel = self.get_kern(fixed_bounds=fixed_bounds)
        kwargs['batch_size'] = kwargs.get('batch_size', 7)
        gpr = CupyNngpr(kernel=kernel, num_nn=num_nn, **kwargs)
        return gpr


if __name__ == '__main__':
    unittest.main()
