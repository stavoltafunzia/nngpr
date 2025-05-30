import unittest
import itertools

import numpy as np
from sklearn.gaussian_process import kernels

from nngpr import batched_kernels
from nngpr.numpy_nngpr import NumpyProxy


class TestBatchedKernels(unittest.TestCase):

    def __init__(self, *args, xp=NumpyProxy(), **kwargs):
        super().__init__(*args, **kwargs)
        rng = np.random.default_rng(290835)
        self.x = rng.random((10, 3))
        self.y = rng.random((5, 3))
        if type(xp) is not list:
            xp = [xp]
        self.xp = xp

    @staticmethod
    def get_all_kernels():
        return [
            (batched_kernels.ConstantKernel(3.1), kernels.ConstantKernel(3.1)), 
            (batched_kernels.WhiteKernel(3.2), kernels.WhiteKernel(3.2)),
            (batched_kernels.RBF(3.3), kernels.RBF(3.3)),
            (batched_kernels.Matern(1.5), kernels.Matern(1.5)),
        ] 

    def perform_test_on_kernel(self, k_in, k_ref):
        for eval_grad, xp in itertools.product([False, True], self.xp):
            with self.subTest(eval_grad=eval_grad, xp=xp):
                k = k_in.bind_to_array_module(xp)
                x, y = xp.from_numpy(self.x), xp.from_numpy(self.y)
                # Test that gives the same result as the scikit learn kernel
                res = k(x, eval_gradient=eval_grad)
                res_ref = k_ref(self.x, eval_gradient=eval_grad)
                grad, grad_ref = None, None
                if eval_grad:
                    self.assertTrue(type(res[1]) is type(x))
                    self.assertTrue(np.allclose(xp.to_numpy(res[1]), res_ref[1]))
                    grad, grad_ref = res[1], res_ref[1]
                    res, res_ref = res[0], res_ref[0]
                self.assertTrue(type(res) is type(x))
                res = xp.to_numpy(res)
                self.assertTrue(np.allclose(res, res_ref))
        
                # Test that works with batched input
                batch_shape = (2,3,4)
                new_x = xp.tile(x, (*batch_shape, *(1 for _ in range(self.x.ndim))))
                new_res = k(new_x, eval_gradient=eval_grad)
                if eval_grad:
                    self.assertEqual(new_res[1].shape, (*batch_shape, *grad.shape))
                    self.assertTrue(np.allclose(xp.to_numpy(new_res[1]), grad_ref[None, None, None, ...]))
                    new_res = new_res[0]
                self.assertEqual(new_res.shape, (*batch_shape, *res.shape))
                self.assertTrue(np.allclose(xp.to_numpy(new_res), res_ref[None, None, None, ...]))

                # Test that it returns float32 when used with float32 input
                tmp = k(xp.from_numpy(self.x.astype('float32')), eval_gradient=eval_grad)
                if eval_grad:
                    self.assertTrue(type(tmp[1]) is type(x))
                    self.assertEqual(xp.to_numpy(tmp[0]).dtype, np.float32)
                    self.assertEqual(xp.to_numpy(tmp[1]).dtype, np.float32)
                    continue
                tmp = xp.to_numpy(tmp)
                self.assertEqual(tmp.dtype, np.float32)

                # Test also the diagonal
                diag_res_ref = k_ref.diag(self.x)
                diag_res = k.diag(x)
                self.assertTrue(type(diag_res) is type(x))
                diag_res = xp.to_numpy(diag_res)
                self.assertEqual(diag_res.shape, diag_res_ref.shape)
                self.assertTrue(np.allclose(diag_res, diag_res_ref))
                diag_res = xp.to_numpy(k.diag(new_x))
                self.assertEqual(diag_res.shape, (*batch_shape, *diag_res_ref.shape))
                self.assertTrue(np.allclose(diag_res, diag_res_ref[None, None, None, ...]))

                res = xp.to_numpy(k.diag(xp.from_numpy(self.x.astype('float32'))))
                self.assertEqual(res.dtype, np.float32)
                
                # Same tests also with y
                res = k(x, y, eval_gradient=eval_grad)
                res_ref = k_ref(self.x, self.y, eval_gradient=eval_grad)
                self.assertTrue(type(res) is type(x))
                res = xp.to_numpy(res)
                self.assertTrue(np.allclose(res, res_ref))

                new_y = xp.tile(y, (*batch_shape, *(1 for _ in range(self.y.ndim))))
                res = k(new_x, new_y, eval_gradient=eval_grad)
                self.assertEqual(res.shape, (*batch_shape, *res_ref.shape))
                res = xp.to_numpy(res)
                self.assertTrue(np.allclose(res, res_ref[None, None, None, ...]))

                res = xp.to_numpy(k(xp.from_numpy(self.x.astype('float32')), 
                                      xp.from_numpy(self.y.astype('float32')), 
                                      eval_gradient=eval_grad))
                self.assertEqual(res.dtype, np.float32)

    def test_constant(self):
        k = batched_kernels.ConstantKernel(3)
        k_ref = kernels.ConstantKernel(3)
        self.perform_test_on_kernel(k, k_ref)

    def test_white(self):
        k = batched_kernels.WhiteKernel(3.3)
        k_ref = kernels.WhiteKernel(3.3)
        self.perform_test_on_kernel(k, k_ref)

    def test_rbf(self):
        for length_scale in [3.4, 6.5 * np.ones(self.x.shape[1])]:
            with self.subTest(anisotrpic=type(length_scale) is np.ndarray): 
                k = batched_kernels.RBF(length_scale)
                k_ref = kernels.RBF(length_scale)
                self.perform_test_on_kernel(k, k_ref)

    def test_matern(self):
        for nu, length_scale in itertools.product([0.5, 1.5, 2.5, np.inf], [3.5, 6.7 * np.ones(self.x.shape[1])]):
            with self.subTest(nu=nu, anisotrpic=type(length_scale) is np.ndarray):
                k = batched_kernels.Matern(length_scale, nu=nu)
                k_ref = kernels.Matern(length_scale, nu=nu)
                self.perform_test_on_kernel(k, k_ref)

    def test_sum(self):
        kernels = self.get_all_kernels()
        for k1, k2 in itertools.product(kernels, kernels):
            k = k1[0] + k2[0]
            k_ref = k1[1] + k2[1]
            self.perform_test_on_kernel(k, k_ref)

    def test_product(self):
        kernels = self.get_all_kernels()
        for k1, k2 in itertools.product(kernels, kernels):
            k = k1[0] * k2[0]
            k_ref = k1[1] * k2[1]
            self.perform_test_on_kernel(k, k_ref)


if __name__ == '__main__':
    unittest.main()
