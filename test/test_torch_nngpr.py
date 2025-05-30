import unittest

import torch
from nngpr.torch_nngpr import TorchWorker, TorchNngpr, TorchProxy

from test.nngpr_base_tester import NngprTester, WorkerTester
from test.test_batched_kernels import TestBatchedKernels


class TestBatchedKernelsTorch(TestBatchedKernels):

    def __init__(self, *args, **kwargs):
        xps = [TorchProxy('cpu')]
        if torch.cuda.is_available():
            xps.append(TorchProxy('cuda:0'))
        super().__init__(*args, xp=xps, **kwargs)


class TestTorchWorker(WorkerTester, unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(TorchWorker('cuda:0' if torch.cuda.is_available() else 'cpu', 
                                    self.get_kern(), accelerate_kernel=False), *args, **kwargs)


class TestTorchWorkerAcc(WorkerTester, unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(TorchWorker('cuda:0' if torch.cuda.is_available() else 'cpu', 
                                    self.get_kern(), accelerate_kernel=True and torch.cuda.is_available()), *args, **kwargs)


class TestTorchNngpr(NngprTester, unittest.TestCase):

    def get_gpr(self, fixed_bounds=False, kernel=None, num_nn=8, **kwargs):
        """Helper function that returns the NNGPR instance"""
        if kernel is None:
            kernel = self.get_kern(fixed_bounds=fixed_bounds)
        kwargs['batch_size'] = kwargs.get('batch_size', 7)
        gpr = TorchNngpr(['cuda:0' if torch.cuda.is_available() else 'cpu'], kernel=kernel, num_nn=num_nn, **kwargs)
        return gpr


if __name__ == '__main__':
    unittest.main()
