import unittest
import itertools
from abc import abstractmethod

import numpy as np
from sklearn import gaussian_process as skgp


class NNGPRTester:

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

    @abstractmethod
    def get_gpr(self, **kwargs):
        pass

    def get_kern(self, fixed_bounds: bool = False):
        """Helper function that returns the kernel"""
        if fixed_bounds:
            return skgp.kernels.ConstantKernel(1, constant_value_bounds="fixed") * skgp.kernels.RBF(
                length_scale=1, length_scale_bounds="fixed") + skgp.kernels.WhiteKernel(
                1, noise_level_bounds="fixed")
        return skgp.kernels.ConstantKernel(1) * skgp.kernels.RBF(length_scale=1) + skgp.kernels.WhiteKernel(1)

    def test_runs(self):
        """Tests that it runs with no errors"""
        for nn_type in ['kernel-space', 'input-space']:
            with self.subTest(nn_type=nn_type):
                gpr = self.get_gpr(nn_type=nn_type)
                gpr.predict(self._x_eval, return_std=True)
                gpr.predict(self._x_eval, return_cov=True)
                gpr.sample_y(self._x_eval, n_samples=2)
                gpr.fit(self._x, self._y)
                gpr.predict(self._x, return_std=True)
                gpr.predict(self._x_eval, return_std=True)
                gpr.predict(self._x, return_cov=True)
                gpr.predict(self._x_eval, return_cov=True)
                gpr.sample_y(self._x_eval, n_samples=2)

    def test_same_lkl(self):
        """Tests that NNGPR gives the same likelihood as the scikit-learn Gaussian Process Regressor when the number of nearest neighbors
        is large enough"""
        theta = np.ones(3) * 0.5
        for y_dim in [1, 100]:
            with self.subTest(y_dim=y_dim):
                mdl_ref = skgp.GaussianProcessRegressor(kernel=self.get_kern(), normalize_y=True)
                mdl_ref.X_train_, mdl_ref.y_train_, mdl_ref.kernel_ = self._x, self._y[:, :y_dim], self.get_kern()
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
                mdl_ref = skgp.GaussianProcessRegressor(kernel=self.get_kern(), normalize_y=True)
                mdl_ref.fit(self._x, self._y[:, :y_dim])
                gpr = self.get_gpr(
                    kernel=self.get_kern(), normalize_y=True, num_nn=self._x.shape[0])
                gpr.fit(self._x, self._y[:, :y_dim])
                delta = np.max(np.abs(mdl_ref.kernel_.theta / gpr.kernel_.theta - 1))
                self.assertLess(delta, 1e-9)

    def test_prior_predict_same_results(self):
        """Tests that NNGPR gives the same prior predictions as the scikit-learn Gaussian Process Regressor when the number of nearest
        neighbors is large enough"""
        mdl_ref = skgp.GaussianProcessRegressor(kernel=self.get_kern(), normalize_y=True)
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

                mdl_ref = skgp.GaussianProcessRegressor(kernel=self.get_kern(fixed_bounds=True), normalize_y=True)
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
                mdl_ref = skgp.GaussianProcessRegressor(kernel=self.get_kern(fixed_bounds=True), normalize_y=True)
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
        n_samples = 10000
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
                mdl_ref = skgp.GaussianProcessRegressor(kernel=self.get_kern(fixed_bounds=True), normalize_y=True)
                mdl_ref.fit(self._x, self._y[:, :y_dim])
                samples_ref = mdl_ref.sample_y(x_eval, n_samples=n_samples, random_state=random_state)
                mean_ref, std_ref = mdl_ref.predict(x_eval, return_std=True)
                random_state += 1

                self.assertEqual(samples.shape, samples_ref.shape)
                mean_delta = np.max(np.abs(np.mean(samples_ref, axis=-1) - mean_ref))
                self.assertLess(np.max(np.abs(np.mean(samples, axis=-1) - mean_pred)), mean_delta * 2)
                std_delta = np.max(np.abs(np.std(samples_ref, axis=-1) - std_ref))
                self.assertLess(np.max(np.abs(np.std(samples, axis=-1) - std_pred)), std_delta * 2)


if __name__ == '__main__':
    unittest.main()
