import itertools
from abc import abstractmethod, ABC

import numpy as np
from sklearn import gaussian_process as skgp

from nngpr.base_nngpr import NngprWorker
from nngpr import batched_kernels
from nngpr.numpy_nngpr import NumpyProxy


class WorkerTester(ABC):

    __test__ = False

    def __init__(self, worker, *args, **kwargs):
        if type(worker) is not list:
            worker = [worker]
        self.workers = worker
        super().__init__(*args, **kwargs)

    def get_kern(self):
        return batched_kernels.RBF(length_scale=1.0)

    def test_empty(self):
        for worker, shape, dtype in itertools.product(self.workers, [2, (5, 6)], [np.float32, np.float64]):
            xp = worker.get_array_module()
            with self.subTest(worker=worker, shape=shape, dtype=dtype):
                xp.empty(shape, dtype=dtype)
            
    def test_get_array_module(self):
        for worker in self.workers:
            with self.subTest(worker=worker):
                xp = worker.get_array_module()
                x1 = xp.empty((5, 2, 3), dtype=np.float32)
                x2 = xp.empty((10, 2, 3), dtype=np.float32)
                self.assertTrue(xp.moveaxis(x1, 0, 2).shape == (2, 3, 5))
                self.assertTrue(xp.swapaxes(x2, 1, 2).shape == (10, 3, 2))
                xp.log(x1 + 2)
                xp.sqrt(x2)
                xp.arange(10)
                xp.where(x1 > 2)
                xp.sum(x1)
                xp.tile(x1, (3, 1, 1))
                x2 = xp.ones((10, 3, 4, 4), dtype=np.float32)
                x2 @ x2

    def test_to_float32(self):
        for worker in self.workers:
            with self.subTest(worker=worker):
                xp = worker.get_array_module()
                x1 = xp.ones((5, 2, 3), dtype=np.float32)
                x2 = xp.ones((10, 2, 3), dtype=np.float64)
                self.assertTrue(x1 is xp.to_float32(x1))
                self.assertFalse(x2 is xp.to_float32(x2))

    def test_to_numpy(self):
        for worker in self.workers:
            with self.subTest(worker=worker):
                xp = worker.get_array_module()
                x1 = xp.empty(3, dtype=np.float32)
                self.assertTrue(type(xp.to_numpy(x1)) is np.ndarray)

    def test_from_numpy(self):
        x = np.ones(6)
        for worker in self.workers:
            with self.subTest(worker=worker):
                xp = worker.get_array_module()
                x1 = xp.empty(3, dtype=np.float32)
                self.assertTrue(type(xp.from_numpy(x)) is type(x1))
        
    def test_find_nearest_neighbors(self):
        for worker, nn_type, ref_cut_index in itertools.product(self.workers, ['kernel-space', 'input-space'], [None, 0, 2, -1]):
            with self.subTest(worker=worker, nn_type=nn_type, ref_cut_index=ref_cut_index):
                xp = worker.get_array_module()
                ref = xp.ones((10, 3))
                query = xp.ones((4, 3)) + 2
                k = self.get_kern()
                res = worker.find_nearest_neighbors(ref, query, nn_type, k, 2, ref_cut_index=ref_cut_index)
                self.assertEqual(type(res), type(ref))

    def test_chofactor_and_chosolve(self):
        x = np.random.rand(3, 5, 5)
        x = x @ np.swapaxes(x, -1, -2) + np.eye(x.shape[-1])[None, ...]
        y = np.random.rand(3, 5, 8)
        for worker in self.workers:
            with self.subTest(worker=worker):
                xp = worker.get_array_module()
                L, L_lower = worker.batched_chofactor(xp.from_numpy(x))
                res = worker.batched_chosolve(L, xp.from_numpy(y), L_lower)
                self.assertEqual(type(res), type(xp.from_numpy(x)))
                self.assertTrue(np.allclose(xp.to_numpy(res), np.linalg.solve(x, y)))

    def test_fill_nn_kernel(self):
        for worker, eval_gradient in itertools.product(self.workers, [False, True]):
            with self.subTest(worker=worker, eval_gradient=eval_gradient):
                xp = worker.get_array_module()
                x = xp.zeros((10, 5), dtype=np.float32)
                nn_indices = np.random.randint(0, x.shape[0], (3, 4), dtype=np.int32)
                k = self.get_kern()
                res = worker.fill_nn_kernel(xp.from_numpy(x), xp.from_numpy(nn_indices), k, eval_gradient=eval_gradient)
                self.assertEqual(type(res[0]), type(x))
                self.assertEqual(type(res[1]), type(x))

    def test_random_normal(self):
        mu = np.ones((3, 1, 2, 1))
        sigma = np.ones((1, 5, 2, 1))
        shape = (3, 5, 2, 10)
        for worker in self.workers:
            with self.subTest(worker=worker):
                xp = worker.get_array_module()
                for random_state in [None, 0, 324]:
                    s = worker.check_random_state(None)
                    res = worker.random_normal(s, mu, sigma, shape)
                    self.assertEqual(type(res), type(xp.from_numpy(mu)))


class NngprTester:

    def __init__(self, *args, **kwargs):
        # Generate some fake data
        rng = np.random.RandomState(234058)
        n = 150
        nj = 4
        ny = 3
        self._x = rng.rand(n, nj)
        self._x_eval = rng.rand(50, nj)
        beta = rng.rand(self._x.shape[1], ny)
        beta[:, -1] += 1
        self._y = self._x @ beta + rng.rand(self._x.shape[0], ny)
        self._y[:, 0] *= 10
        super().__init__(*args, **kwargs)
        self.np = NumpyProxy()

    @abstractmethod
    def get_gpr(self, **kwargs):
        pass

    def get_kern(self, fixed_bounds: bool = False):
        """Helper function that returns the kernel"""
        if fixed_bounds:
            return batched_kernels.ConstantKernel(1, constant_value_bounds="fixed") * batched_kernels.RBF(
                length_scale=1, length_scale_bounds="fixed") + batched_kernels.WhiteKernel(
                1, noise_level_bounds="fixed")
        return batched_kernels.ConstantKernel(1) * batched_kernels.RBF(length_scale=1) + batched_kernels.WhiteKernel(1)


    def test_runs(self):
        """Tests that it runs with no errors"""

        def run(nn_type=None, distribute_method=None):
            gpr = self.get_gpr(nn_type=nn_type, distribute_method=distribute_method)
            if (len(gpr.workers) == 1) and (distribute_method is not None):
                return
            gpr.predict(self._x_eval, return_std=True)
            gpr.predict(self._x_eval, return_cov=True)
            gpr.sample_y(self._x_eval, n_samples=2)
            gpr.fit(self._x, self._y)
            gpr.predict(self._x)
            gpr.predict(self._x, return_std=True)
            gpr.predict(self._x_eval, return_std=True)
            gpr.predict(self._x, return_cov=True)
            gpr.predict(self._x_eval, return_cov=True)
            gpr.sample_y(self._x_eval, n_samples=2)

        for nn_type in ['kernel-space', 'input-space']:
            with self.subTest(nn_type=nn_type):
                run(nn_type=nn_type)

        for distribute_method in ['joblib', 'multiprocessing', 'multiprocessing-heterogenous', 'sequential']:
            with self.subTest(distribute_method=distribute_method):
                run(distribute_method=distribute_method)

    def test_same_lkl(self):
        """Tests that NNGPR gives the same likelihood as the scikit-learn Gaussian Process Regressor when the number of nearest neighbors
        is large enough"""
        kernel = self.get_kern()
        theta = np.ones(len(kernel.theta)) * 0.5
        for y_dim in [1, 100]:
            with self.subTest(y_dim=y_dim):
                mdl_ref = skgp.GaussianProcessRegressor(kernel=self.get_kern().bind_to_array_module(self.np), normalize_y=True)
                mdl_ref.X_train_, mdl_ref.y_train_, mdl_ref.kernel_ = self._x, self._y[:, :y_dim], self.get_kern().bind_to_array_module(self.np)
                lkl_ref, grad_ref = mdl_ref.log_marginal_likelihood(theta, eval_gradient=True)
                gpr = self.get_gpr(normalize_y=True, num_nn=self._x.shape[0])
                gpr.X_train_, gpr.y_train_, gpr.kernel_ = self._x, self._y[:, :y_dim], self.get_kern()
                lkl, grad = gpr.log_marginal_likelihood(theta, eval_gradient=True)

                self.assertEqual(grad.shape, grad.shape)
                delta = np.max(np.abs(lkl / lkl_ref - 1))
                self.assertLess(delta, 1e-10)
                delta = np.max(np.abs(grad / grad_ref - 1))
                self.assertLess(delta, 1e-10)

    def test_same_fit_results(self):
        """Tests that NNGPR.fit gives the same likelihood as the scikit-learn Gaussian Process Regressor fit when the number of nearest
        neighbors is large enough"""
        for y_dim in [1, 100]:
            with self.subTest(y_dim=y_dim):
                mdl_ref = skgp.GaussianProcessRegressor(kernel=self.get_kern().bind_to_array_module(self.np), normalize_y=True)
                mdl_ref.fit(self._x, self._y[:, :y_dim])
                gpr = self.get_gpr(
                    kernel=self.get_kern(), normalize_y=True, num_nn=self._x.shape[0])
                gpr.fit(self._x, self._y[:, :y_dim])
                delta = np.max(np.abs(mdl_ref.kernel_.theta / gpr.kernel_.theta - 1))
                self.assertLess(delta, 1e-9)

    def test_prior_predict_same_results(self):
        """Tests that NNGPR gives the same prior predictions as the scikit-learn Gaussian Process Regressor when the number of nearest
        neighbors is large enough"""
        mdl_ref = skgp.GaussianProcessRegressor(kernel=self.get_kern().bind_to_array_module(self.np), normalize_y=True)
        mean_ref, std_ref = mdl_ref.predict(self._x, return_std=True)
        _, cov_ref = mdl_ref.predict(self._x, return_cov=True)
        for conditioning_method in ['only-train', 'full', 'full-double-nn']:
            with self.subTest(conditioning_method=conditioning_method):
                gpr = self.get_gpr(
                    fixed_bounds=True, normalize_y=True,
                    num_nn=self._x.shape[0])
                _, cov = gpr.predict(self._x, return_cov=True, conditioning_method=conditioning_method)
                mean, std = gpr.predict(self._x, return_std=True, conditioning_method=conditioning_method)
                delta = mean - mean_ref
                self.assertLess(np.max(np.abs(delta)), 1e-10)
                delta = std - std_ref
                self.assertLess(np.max(np.abs(delta)), 1e-10)
                if conditioning_method != 'only-train':
                    delta = cov - cov_ref
                    self.assertLess(np.max(np.abs(delta)), 1e-10)

    def test_predict_same_results(self):
        """Tests that NNGPR gives the same posterior predictions as the scikit-learn Gaussian Process Regressor when the number of nearest
        neighbors is large enough"""
        for conditioning_method, use_eval, y_dim in itertools.product(
                ['only-train', 'full', 'full-double-nn'], [True, False], [1, 100]):
            with self.subTest(conditioning_method=conditioning_method, use_eval=use_eval, y_dim=y_dim):
                x_eval = self._x_eval if use_eval else self._x

                mdl_ref = skgp.GaussianProcessRegressor(kernel=self.get_kern(fixed_bounds=True).bind_to_array_module(self.np), normalize_y=True)
                mdl_ref.fit(self._x, self._y[:, :y_dim])
                mean_ref, std_ref = mdl_ref.predict(x_eval, return_std=True)
                _, cov_ref = mdl_ref.predict(x_eval, return_cov=True)

                gpr = self.get_gpr(
                    fixed_bounds=True, normalize_y=True,
                    num_nn=self._x.shape[0] + x_eval.shape[0] if conditioning_method == 'full' else self._x.shape[0])
                gpr.fit(self._x, self._y[:, :y_dim])
                _, cov = gpr.predict(x_eval, return_cov=True, conditioning_method=conditioning_method)
                mean, std = gpr.predict(x_eval, return_std=True, conditioning_method=conditioning_method)

                self.assertEqual(mean.shape, mean_ref.shape)
                self.assertEqual(std.shape, std_ref.shape)
                self.assertEqual(cov.shape, cov_ref.shape)
                delta = mean - mean_ref
                self.assertLess(np.max(np.abs(delta)), 1e-10)
                delta = std - std_ref
                self.assertLess(np.max(np.abs(delta)), 1e-10)
                if conditioning_method != 'only-train':
                    delta = cov - cov_ref
                    self.assertLess(np.max(np.abs(delta)), 1e-10)

    def test_prior_sample(self):
        """Tests that NNGPR prior samples are consistent with predictions and with the prior samples from the scikit-learn Gaussian
        Process Regressor when the number of nearest neighbors is large enough"""
        n_samples = 10000
        random_state = 1
        for conditioning_method in ['only-train', 'full', 'full-double-nn']:
            with self.subTest(conditioning_method=conditioning_method):
                gpr = self.get_gpr(
                    fixed_bounds=True, normalize_y=True,
                    num_nn=self._x.shape[0])
                samples = gpr.sample_y(self._x, n_samples=n_samples, random_state=random_state,
                                       conditioning_method=conditioning_method)
                mean_pred, std_pred = gpr.predict(
                    self._x, conditioning_method=conditioning_method, return_std=True)
                mdl_ref = skgp.GaussianProcessRegressor(kernel=self.get_kern(fixed_bounds=True).bind_to_array_module(self.np), normalize_y=True)
                samples_ref = mdl_ref.sample_y(self._x, n_samples=n_samples, random_state=random_state)
                mean_ref, std_ref = mdl_ref.predict(self._x, return_std=True)
                random_state += 1
                mean_delta = np.max(np.abs(np.mean(samples_ref, axis=1) - mean_ref))
                self.assertLess(np.max(np.abs(np.mean(samples, axis=1) - mean_pred)), mean_delta * 2)
                std_delta = np.max(np.abs(np.std(samples_ref, axis=1) - std_ref))
                self.assertLess(np.max(np.abs(np.std(samples, axis=1) - std_pred)), std_delta * 2)

    def test_sample(self):
        """Tests that NNGPR posterior samples are consistent with predictions and with the posterior samples from the scikit-learn Gaussian
        Process Regressor when the number of nearest neighbors is large enough"""
        n_samples = 1000 # int(2 * 1024**3 / self._x.shape[0] / 8) #10000
        random_state = 1
        for conditioning_method, use_eval, y_dim in itertools.product(
                ['only-train', 'full', 'full-double-nn'], [False, True], [1, 100]):
            with self.subTest(
                    conditioning_method=conditioning_method, use_eval=True, y_dim=y_dim):
                x_eval = self._x_eval if use_eval else self._x
                gpr = self.get_gpr(
                    fixed_bounds=True, normalize_y=True,
                    num_nn=self._x.shape[0])
                gpr.fit(self._x, self._y[:, :y_dim])
                samples = gpr.sample_y(x_eval, n_samples=n_samples, random_state=random_state,
                                       conditioning_method=conditioning_method)
                mean_pred, std_pred = gpr.predict(
                    x_eval, conditioning_method=conditioning_method, return_std=True)
                mdl_ref = skgp.GaussianProcessRegressor(kernel=self.get_kern(fixed_bounds=True).bind_to_array_module(self.np), normalize_y=True)
                mdl_ref.fit(self._x, self._y[:, :y_dim])
                samples_ref = mdl_ref.sample_y(x_eval, n_samples=n_samples, random_state=random_state)
                mean_ref, std_ref = mdl_ref.predict(x_eval, return_std=True)
                random_state += 1

                self.assertEqual(samples.shape, samples_ref.shape)
                mean_delta = np.max(np.abs(np.mean(samples_ref, axis=-1) - mean_ref))
                self.assertLess(np.max(np.abs(np.mean(samples, axis=-1) - mean_pred)), mean_delta * 2)
                std_delta = np.max(np.abs(np.std(samples_ref, axis=-1) - std_ref))
                self.assertLess(np.max(np.abs(np.std(samples, axis=-1) - std_pred)), std_delta * 2)

    def test_large_sample_size(self):
        """Tests that Nngpr sample_y works on large sizes"""
        
        size_gb = 2
        random_state = 1
        x = self._x
        gpr = self.get_gpr(fixed_bounds=True, normalize_y=True, num_nn=32)
        n_samples = int(size_gb * 1024**3 / self._x.shape[0] / 8)
        samples = gpr.sample_y(x, n_samples=n_samples, random_state=random_state, 
                               conditioning_method='only-train')
        self.assertEqual(samples.shape, (x.shape[0], n_samples))

        random_state = 2
        x = np.concatenate([self._x] * int(n_samples / self._x.shape[0] / 10), axis=0)
        gpr = self.get_gpr(fixed_bounds=True, normalize_y=True, num_nn=32)
        n_samples = int(2 * 1024**3 / x.shape[0] / 8)
        samples = gpr.sample_y(x, n_samples=n_samples, random_state=random_state, 
                               conditioning_method='only-train')
        self.assertEqual(samples.shape, (x.shape[0], n_samples))
        