from operator import itemgetter
import numbers
from numbers import Integral

import numpy as np
import cupy as cp
import cupyx as cpx
from cupy_backends.cuda.libs import cublas, cusolver
from cupy.cuda import device
from scipy import linalg as sl
from scipy import sparse as sp
from sklearn.gaussian_process import GaussianProcessRegressor as SKGaussianProcessRegressor
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.parallel import Parallel, delayed
from sklearn.gaussian_process import kernels
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn import base as sk_base


class CUNNGPR(SKGaussianProcessRegressor):
    """Implements Nearest Neighbor Gaussian Process Regressor (cunngpr.cunngpr.CUNNGPR) using Cuda to accelerate calculations.
    Currently supports only kernels of the type WhiteKernel + ConstantKernel * RBF.

    It is built on top of sklearn GaussianProcessRegressor, maintaining the same api.

    Parameters
    ----------
    kernel: see sklearn.gaussian_process.GaussianProcessRegressor

    alpha : see sklearn.gaussian_process.GaussianProcessRegressor

    optimizer : see sklearn.gaussian_process.GaussianProcessRegressor

    n_restarts_optimizer : see sklearn.gaussian_process.GaussianProcessRegressor

    normalize_y : see sklearn.gaussian_process.GaussianProcessRegressor

    copy_X_train : see sklearn.gaussian_process.GaussianProcessRegressor

    n_targets : see sklearn.gaussian_process.GaussianProcessRegressor

    random_state : see sklearn.gaussian_process.GaussianProcessRegressor

    num_nn : int, default 32
        Number of nearest neighbors to use.

    n_jobs : int | None, default=None
        The number of parallel jobs to run for fit, predict or sampling. None means 1 unless in a joblib.parallel_backend context. -1 means
        using all processors.

    nn_type : str, default 'kernel-space'
        Search space for the nearest neighbors. Can be either 'kernel-space' or 'input-space'. If 'kernel-space' nearest neighbors
        are searched in the kernel space, i.e. the neighbors of a query point are the points with the highest covariance w.r.t. the 
        query point. When 'input-space' nearest neighbors are searched in the input feature space, using euclidean distance.

    batch_size : int or None, default None
        Batch size used to split the calculation in batches. Large batch size may cause out of memory errors. Low batch sizes may prevent
        full GPU resources exploitation. If None it's set on simple heuristic rule.

    allow_downcast_f32 : str, default 'only-nn'
        Specifies if input arrays can be downcasted to float32, as some cunsomers GPUs have crippled float64 performance. Must be 'no', 
        'yes' or 'only-nn'. If 'no', no downcast is performed. When 'only-nn', downcast is done only for the nearest neighbors search.
        If 'yes', then downcast is done for all operations (including matrix inverse operations). When 'yes', this may negatively affect
        convergence of the optimisation problem when fitting. 


    Attributes
    ----------
    X_train_ : see sklearn.gaussian_process.GaussianProcessRegressor

    y_train_ : see sklearn.gaussian_process.GaussianProcessRegressor

    kernel : see sklearn.gaussian_process.GaussianProcessRegressor

    num_nn : int
        Number of nearest neighbors

    n_jobs : int | None.
        Number of parallel jobs to run for fit, predict or sampling

    allow_downcast_f32 : str, default 'only-nn'
        Specifies if input arrays can be downcasted to float32, as some cunsomers GPUs have crippled float64 performance. Must be 'no', 
        'yes' or 'only-nn'. If 'no', no downcast is performed. When 'only-nn', downcast is done only for the nearest neighbors search.
        If 'yes', then downcast is done for all operations (including matrix inverse operations). When 'yes', this may negatively affect
        convergence of the optimisation problem when fitting.

    """

    _parameter_constraints: dict = {
        **SKGaussianProcessRegressor._parameter_constraints,
        "num_nn": [Interval(Integral, 1, None, closed="left")],
        "n_jobs": [Integral, None],
        "nn_type": [StrOptions({"kernel-space", "input-space"})],
        "batch_size": [Interval(Integral, 1, None, closed="left"), None],
        "allow_downcast_f32": [StrOptions({"yes", "no", "only-nn"})],
    }

    def __init__(
            self,
            kernel=None,
            *,
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=0,
            normalize_y=False,
            copy_X_train=True,
            random_state=None,
            num_nn=32,
            n_jobs=None,
            nn_type='kernel-space',
            batch_size=None,
            allow_downcast_f32='only-nn'):

        super().__init__(
            kernel=kernel, alpha=alpha, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y, copy_X_train=copy_X_train, random_state=random_state)

        # Store inputs
        self.num_nn = num_nn
        self.n_jobs = n_jobs
        self.nn_type = nn_type
        if batch_size is None:
            batch_size = max(1, min(10000, int(0.8 * 1024**3 / (4 if allow_downcast_f32 =='yes' else 8) / num_nn **2)))
        self.batch_size = batch_size
        self.allow_downcast_f32 = allow_downcast_f32

    @sk_base._fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit Gaussian process regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            NNGPR class instance.
        """

        if self.kernel is None:  # Use the default kernel
            self.kernel_ = self.get_default_kernel()
        else:
            self.kernel_ = sk_base.clone(self.kernel)

        self._rng = check_random_state(self.random_state)

        if self.kernel_.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False
        X, y = validate_data(self, X, y, multi_output=True, y_numeric=True, ensure_2d=ensure_2d, dtype=dtype)

        n_targets_seen = y.shape[1] if y.ndim > 1 else 1
        if self.n_targets is not None and n_targets_seen != self.n_targets:
            raise ValueError(
                "The number of targets seen in `y` is different from the parameter "
                f"`n_targets`. Got {n_targets_seen} != {self.n_targets}."
            )

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y, axis=0), copy=False)

            # Remove mean and make unit variance
            y = (y - self._y_train_mean) / self._y_train_std

        else:
            shape_y_stats = (y.shape[1],) if y.ndim == 2 else 1
            self._y_train_mean = np.zeros(shape=shape_y_stats)
            self._y_train_std = np.ones(shape=shape_y_stats)

        if np.iterable(self.alpha):
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError(
                    "alpha must be a scalar or an array with only one element"
                )

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True, clone_kernel=False
                    )
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta, clone_kernel=False)

            # First optimize starting from theta specified in kernel
            optima = [
                (
                    self._constrained_optimization(
                        obj_func, self.kernel_.theta, self.kernel_.bounds
                    )
                )
            ]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite."
                    )
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial, bounds)
                    )
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()

            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(
                self.kernel_.theta, clone_kernel=False
            )

        self._free_cupy_memory()

        return self

    def log_marginal_likelihood(self, theta=None, eval_gradient=False, clone_kernel=True):
        """Return log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like of shape (n_kernel_params,) default=None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default=False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        clone_kernel : bool, default=True
            If True, the kernel attribute is copied. If False, the kernel
            attribute is modified, but may result in a performance improvement.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : ndarray of shape (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """

        if theta is None:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        if clone_kernel:
            kernel = self.kernel_.clone_with_theta(theta)
        else:
            kernel = self.kernel_
            kernel.theta = theta

        out_dtype = cp.float32 if self.allow_downcast_f32 == 'yes' else cp.float64

        # Define x train
        x_full = self.X_train_
        nt = x_full.shape[0]
        n_theta = len(theta)

        # get kernel params
        kernel_params = self.get_kernel_params(kernel)
        theta_mapping = self.get_kernel_theta_mapping(kernel)
        if any(np.any(np.isnan(v)) for k, v in theta_mapping.items()):
            if eval_gradient:
                return np.nan, np.ones(n_theta) * np.nan
            return np.nan
        
        # Support multi-dimensional output of self.y_train_
        y_train_full = self.y_train_
        if y_train_full.ndim == 1:
            y_train_full = y_train_full[:, np.newaxis]

        # Define function that performs calculation on a batch
        def run_batch(i_batch):
            i0 = self.batch_size * i_batch
            i1 = min(self.batch_size * (i_batch + 1), nt)

            # Find nearest neighbours
            nn_indices = self.find_nearest_neighbours(
                kernel_params, min(self.num_nn, i1 - 1), x_full[:i1], x_full[i0:i1], ref_cut_index=i0)
            nn_indices = cp.concatenate([
                cp.arange(i0, i0 + nn_indices.shape[0]).reshape((-1, 1)), nn_indices], axis=1)

            # Evaluates the kernel and kernel gradient
            K, K_gradient = self.fill_nn_kernel(
                x_full, nn_indices, kernel_params, theta_mapping, eval_gradient=eval_gradient)
            if K.dtype != out_dtype:
                K = K.astype(out_dtype)
            if K_gradient.dtype != out_dtype:
                K_gradient = K_gradient.astype(out_dtype)
            
            # Add jitter to the kernel
            ind = cp.where(nn_indices > -1)
            K[ind[0], ind[1], ind[1]] += self.alpha
            if eval_gradient:
                K_gradient = cp.moveaxis(K_gradient, -1, 0)  # Move the axis corresponding to theta at the beginning

            n = K.shape[0]

            # Calculate the Cholesky decomposition
            L = cp.linalg.cholesky(K[:, 1:, 1:])

            # Calculate the y train
            y_train = cp.array(y_train_full[i0:i1])
            y_train_nn = self._build_ytrain_given_nn_indices(y_train_full, nn_indices)
            del nn_indices

            # Now calculate the log marginal likelihood

            # Define matrices K_xn and K_nn_inv
            K_xn = K[:, 0, 1:].reshape((n, 1, K.shape[1] - 1))

            def K_nn_inv(right, add_dim=False):
                this_l = L
                if add_dim:
                    this_l = L[cp.newaxis, :]
                shape = cp.broadcast_shapes(this_l.shape[:-2], right.shape[:-2])
                if shape != this_l.shape[:-2]:
                    new_l = cp.empty((*shape, *this_l.shape[-2:]), dtype=this_l.dtype,
                                      order='F' if this_l.flags['F_CONTIGUOUS'] else 'C')
                    new_l[...] = this_l
                    this_l = new_l
                if shape != right.shape[:-2]:
                    new_right = cp.empty((*shape, *right.shape[-2:]), dtype=right.dtype,
                                          order='F' if right.flags['F_CONTIGUOUS'] else 'C')
                    new_right[...] = right
                    right = new_right
                else:
                    right = cp.array(right)  # We need to make a copy to avoid overwriting
                return cusolver_potrs(this_l, right, True)
                
            mu = (K_xn @ K_nn_inv(y_train_nn[:, 1:])).reshape(y_train.shape)
            sigma = cp.sqrt(K[:, 0, 0].reshape((n, 1)) - (K_xn @ K_nn_inv(cp.swapaxes(K_xn, 1, 2))
                                                          ).reshape((n, 1)))
            this_log_lkl = -0.5 * (y_train - mu) ** 2 / sigma ** 2 - 0.5 * np.log(2 * np.pi) - cp.log(sigma)

            # the log likehood is sum-up across the outputs and the first dimension
            log_likelihood = float(this_log_lkl.sum(axis=(0, -1)))

            if eval_gradient:
                # Expand quantities by adding the dimension corresponding to theta
                sigma, mu, y_train = sigma[cp.newaxis, :], mu[cp.newaxis, :], y_train[cp.newaxis, :]

                # Derivative of K_nn
                dK_nn_inv_dtheta = lambda right: -K_nn_inv(
                    K_gradient[:, :, 1:, 1:] @ K_nn_inv(right)[cp.newaxis, :], add_dim=True)
                # Derivative of K_xn
                dK_xn_dtheta = K_gradient[:, :, 0, 1:].reshape((n_theta, n, 1, K.shape[1] - 1))
                # Derivative of mu
                dmu_dtheta = (dK_xn_dtheta @ K_nn_inv(y_train_nn[:, 1:])[cp.newaxis, :]).reshape(
                    (n_theta, *y_train.shape[1:])) + \
                    (K_xn[cp.newaxis, :] @ dK_nn_inv_dtheta(y_train_nn[:, 1:])).reshape((n_theta, *y_train.shape[1:]))

                # Derivarive of sigma
                dsigma_dtheta = 0.5 / sigma * (
                        K_gradient[:, :, 0, 0].reshape((n_theta, n, 1)) -
                        2 * (dK_xn_dtheta @ (K_nn_inv(cp.swapaxes(K_xn, 1, 2)))[cp.newaxis, :]).reshape(
                            (n_theta, n, 1)) - (K_xn[cp.newaxis, :] @ dK_nn_inv_dtheta(cp.swapaxes(
                                K_xn, 1, 2))).reshape((n_theta, n, 1)))

                log_likelihood_gradient = (-1 / sigma + (y_train - mu) ** 2 / sigma ** 3) * dsigma_dtheta + (
                            y_train - mu) / sigma ** 2 * dmu_dtheta

                log_likelihood_gradient = cp.sum(
                    log_likelihood_gradient, axis=(1, 2)).get()  # Axis 0 is the theta parameter, axis 2 is the dimension of the output
            else:
                log_likelihood_gradient = np.zeros(n_theta)
                
            return log_likelihood, log_likelihood_gradient

        num_batches = int(np.ceil(nt / self.batch_size))
        # TODO: could be executed in parallel on multiple GPUs
        batch_results = [run_batch(i) for i in range(num_batches)]
        log_likelihood = sum(x[0] for x in batch_results)
        self._free_cupy_memory()

        if eval_gradient:
            log_likelihood_gradient = np.sum([x[1] for x in batch_results], axis=0)
            return log_likelihood, log_likelihood_gradient

        return log_likelihood

    def sample_y(self, X, n_samples=1, random_state=0, conditioning_method=None):
        """Draw samples from Gaussian process and evaluate at X.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Query points where the GP is evaluated.

        n_samples : int, default=1
            Number of samples drawn from the Gaussian process per query point.

        random_state : int, cp.random.RandomState instance or None, default=0
            Determines random number generation to randomly draw samples.
            Pass an int for reproducible results across multiple function
            calls.
            See :term:`Glossary <random_state>`.

        conditioning_method : str | None, default = 'only-train'. Conditioning method, possible values are: 'only-train', 'full',
            'full-double-nn'. Changes the so-called reference set as in Datta 2016. When 'only-train', the reference set corresponds to the
            training set. When 'full', the reference set corresponds to the training set plus the evaluation set (X). When 'full-double-nn',
            the reference set is as in 'full', however twice the amount of nearest neighbour per each point are used; half of the nearest
             neighbours are taken from the training set, and half from the evaluation set (X).

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples), or \
            (n_samples_X, n_targets, n_samples)
            Values of n_samples samples drawn from Gaussian process and
            evaluated at query points.
        """

        # Check input
        conditioning_method = 'only-train' if conditioning_method is None else conditioning_method
        assert conditioning_method in {'only-train', 'full', 'full-double-nn'}
        if self.kernel is None or self.kernel.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False
        X = validate_data(self, X, ensure_2d=ensure_2d, dtype=dtype, reset=False)
        nq = X.shape[0]

        if not hasattr(self, "X_train_"):  # Unfitted; predict based on GP prior
            x_train = np.empty((0, X.shape[1]), dtype=X.dtype)
            nt = 0
            y_dim = 1
            y_train = np.zeros((0, 1), dtype=X.dtype)
            kernel = self.kernel if self.kernel is not None else self.get_default_kernel()
        else:
            x_train = self.X_train_
            nt = self.X_train_.shape[0]
            y_train = self.y_train_
            if y_train.ndim == 1:
                y_train = y_train[:, np.newaxis]
            y_dim = y_train.shape[-1]
            kernel = self.kernel_

        out_dtype = cp.float32 if self.allow_downcast_f32 == 'yes' else cp.float64

        # get kernel params
        kernel_params = self.get_kernel_params(kernel)
        theta_mapping = self.get_kernel_theta_mapping(kernel)

        # If conditioning_method is only-train, then each sample is independent of the others and we can use
        # a faster (full parallel) algorithm
        if conditioning_method == 'only-train':
            rng = self.check_cupy_random_state(random_state)
            mu, sigma = self.predict(X, return_std=True)
            mu, sigma = cp.array(mu[..., np.newaxis]), cp.array(sigma[..., np.newaxis])
            if y_dim == 1:
                mu, sigma = mu[..., np.newaxis], sigma[..., np.newaxis]
            y_samples = rng.normal(loc=mu, scale=sigma, size=(nq, y_dim, n_samples))
            if y_samples.dtype != out_dtype:
                y_samples = y_samples.astype(out_dtype)
            y_samples = y_samples.get()

            if y_dim == 1:
                y_samples = y_samples[:, 0, :]  # Remove dimension corresponding to the y-dimension

            self._free_cupy_memory()

            return y_samples
        
        # If conditioning_method is not only-train, continue here with the sequential algorithm
        rng = check_random_state(random_state)

        # Find nearest neighbours. They could be searched in batches, but not really needed since the memory footprint is 
        # negligible. The biggest memory usage comes from the kernel K and its Cholesky decomposition.
        nn_indices, x_full = self.find_nn_indices_for_train_and_eval(
            kernel_params, x_train, X, condition_on_eval=conditioning_method != 'only-train',
            double_nn=conditioning_method == 'full-double-nn')
        nn_indices = nn_indices.get()

        # Allocate output and temporary vars
        y_samples = np.ones((nq, y_dim, n_samples)) * np.nan
        y_nn = np.empty((nn_indices.shape[1] - 1, y_dim, n_samples))

        # Loop over batches of data in case the entire arrays cannot be all stored in memory
        for i_batch in range(int(np.ceil(nq / self.batch_size))):
            i0 = self.batch_size * i_batch
            i1 = min(self.batch_size * (i_batch + 1), nq)

            # Evaluates the kernel and kernel gradient
            K, _ = self.fill_nn_kernel(
                x_full, nn_indices[i0:i1], kernel_params, theta_mapping, eval_gradient=False)
            if K.dtype != out_dtype:
                K = K.astype(out_dtype)

            # Add jitter to the kernel
            ind = np.where((nn_indices[i0:i1] < nt) & (nn_indices[i0:i1] >= 0))
            K[ind[0], ind[1], ind[1]] += self.alpha
            del ind

            # Calculate the Cholesky decomposition
            L = cp.linalg.cholesky(K[:, 1:, 1:])

            K, L = K.get(), L.get()

            # Fill output
            for i in range(L.shape[0]):
                assert nn_indices[i + i0, 0] == nt + i0 + i
                this_ind = nn_indices[i + i0, 1:]
                is_neg = this_ind < 0
                is_train = (this_ind < nt) & (this_ind >= 0)
                not_train = this_ind >= nt
                non_train_ind = this_ind[not_train] - nt

                y_nn[is_neg, :, :] = 0
                y_nn[is_train, :, :] = y_train[this_ind[is_train]][:, :, np.newaxis]
                y_nn[not_train, :, :] = y_samples[non_train_ind]

                this_K_xn = K[i, 0, 1:].reshape((1, -1))
                this_K_xn[0, is_neg] = 0
                this_K_nn_inv = lambda right: sl.cho_solve((L[i], True), right, overwrite_b=False)

                if this_K_xn.size > 0:
                    mu = np.einsum('i,ijk->jk', this_K_nn_inv(this_K_xn.T).reshape((-1)),
                                   y_nn)  # k is the sample index, j is the y-dimension index, i is the nn index
                    sigma = max(0, np.sqrt(K[i, 0, 0] - (this_K_xn @ this_K_nn_inv(this_K_xn.T))))  # May be negative due to rounding
                else:
                    mu = 0
                    sigma = np.sqrt(K[i, 0, 0])

                y_samples[i + i0, :, :] = rng.normal(loc=mu, scale=sigma, size=(y_dim, n_samples))

        if hasattr(self, '_y_train_std'):
            y_samples = y_samples * self._y_train_std.reshape((1, -1, 1)) + self._y_train_mean.reshape((1, -1, 1))  # Undo y scaling

        if y_dim == 1:
            y_samples = y_samples[:, 0, :]  # Remove dimension corresponding to the y-dimension

        self._free_cupy_memory()

        return y_samples

    def predict(self, X, return_std=False, return_cov=False, conditioning_method=None):
        """Draw samples from Gaussian process and evaluate at X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated.

        return_std : bool, default=False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_cov : bool, default=False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean.

        conditioning_method : str | None, default = 'only-train'. Conditioning method, possible values are: 'only-train', 'full',
            'full-double-nn'. Changes the so-called reference set as in Datta 2016. When 'only-train', the reference set corresponds to the
            training set. When 'full', the reference set corresponds to the training set plus the evaluation set (X). When 'full-double-nn',
            the reference set is as in 'full', however twice the amount of nearest neighbour per each point are used; half of the nearest
             neighbours are taken from the training set, and half from the evaluation set (X).

        Returns
        -------
        y_mean : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Mean of predictive distribution at query points.

        y_std : ndarray of shape (n_samples,) or (n_samples, n_targets), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_std` is True.

        y_cov : ndarray of shape (n_samples, n_samples) or \
                (n_samples, n_samples, n_targets), optional
            Covariance of joint predictive distribution at query points.
            Only returned when `return_cov` is True. Remark: the covariance matrix of a nearest neighbour gaussian process is still dense!
        """

        # Check input
        conditioning_method = 'only-train' if conditioning_method is None else conditioning_method
        assert conditioning_method in {'only-train', 'full', 'full-double-nn'}
        if self.kernel is None or self.kernel.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False

        # Take some variables for later usage
        X = validate_data(self, X, ensure_2d=ensure_2d, dtype=dtype, reset=False)
        nq = X.shape[0]

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            x_train = np.empty((0, X.shape[1]), dtype=X.dtype)
            y_train = np.empty((0, 1), dtype=X.dtype)
            nt = 0
            is_prior = True
            kernel = self.kernel if self.kernel is not None else self.get_default_kernel()
        else:
            x_train = self.X_train_
            nt = self.X_train_.shape[0]
            y_train = self.y_train_
            is_prior = False
            kernel = self.kernel_

        # get kernel params
        kernel_params = self.get_kernel_params(kernel)
        theta_mapping = self.get_kernel_theta_mapping(kernel)

        # Faster calculation for prior
        if is_prior and not return_cov:
            mean = np.zeros(X.shape[0])
            if return_std:
                std = np.sqrt(kernel.diag(X))
                return mean, std
            return mean

        # Support multi-dimensional output of y_train
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]
        y_dim = y_train.shape[-1]

        # Define some functions to format the output as needed
        def format_mean(mean):
            if hasattr(self, '_y_train_std'):
                y_train_std = self._y_train_std.astype(mean.dtype)
                y_train_mean = self._y_train_mean.astype(mean.dtype)
                mean = y_train_std.reshape((1, -1)) * mean + y_train_mean.reshape((1, -1))  # Undo y scaling
            if mean.shape[1] == 1:  # Squeeze y_dim
                mean = mean[:, 0]
            return mean
        
        def format_sigma(sigma):
            if hasattr(self, '_y_train_std'):
                y_train_std = self._y_train_std.astype(sigma.dtype)
                sigma = sigma.reshape((-1, 1)) * y_train_std.reshape((1, -1))
            if (len(sigma.shape) > 1) and (sigma.shape[1] == 1):
                sigma = sigma[:, 0]
            return sigma

        # If conditioning_method is only-train then each sample is independent of the others. Thus, if return_cov is False, we can use
        # a faster (full parallel) algorithm
        if (conditioning_method == 'only-train') and (not return_cov):

            # Loop over batches of data in case the entire arrays cannot be all stored in memory
            def run_batch(i_batch):
                i0 = self.batch_size * i_batch
                i1 = min(self.batch_size * (i_batch + 1), nq)

                # Find nearest neighbours. They could be searched in batches, however when the conditioning_method is different than 'only_train',
                # there is little gain in memory usage. Moreover, biggest memory usage comes from the kernel K and its Cholesky decomposition.
                nn_indices, x_full = self.find_nn_indices_for_train_and_eval(
                    kernel_params, x_train, X[i0:i1], condition_on_eval=False, double_nn=False)

                # Evaluates the kernel and kernel gradient
                K, _ = self.fill_nn_kernel(
                    x_full, nn_indices, kernel_params, theta_mapping, eval_gradient=False)
                # Add jitter to the kernel
                ind = np.where((nn_indices < nt) & (nn_indices >= 0))
                K[ind[0], ind[1], ind[1]] += self.alpha
                del ind

                # Calculate y_train_nn
                y_train_nn = self._build_ytrain_given_nn_indices(y_train, nn_indices[:, 1:])
                del nn_indices

                # Calculate the Cholesky decomposition
                L = cp.linalg.cholesky(K[:, 1:, 1:])

                # Define relevant matrices
                K_xn = K[:, :1, 1:]
                def K_nn_inv(right):
                    return cusolver_potrs(L, cp.array(right), True)  # We need to make a copy to avoid overwriting
                
                # Calculate mean
                mean = (K_xn @ K_nn_inv(y_train_nn))[:, 0, :].get()

                if return_std:
                    n = i1 - i0
                    std = cp.sqrt(K[:, 0, 0].reshape((n, 1)) - (K_xn @ K_nn_inv(np.swapaxes(K_xn, 1, 2))
                                                          ).reshape((n, 1))).reshape(-1).get()
                else:
                    std = None

                return mean, std

            num_batches = int(np.ceil(nq / self.batch_size))
            batch_results = Parallel(self.n_jobs)(delayed(run_batch)(i) for i in range(num_batches))
            self._free_cupy_memory()
            mean = np.concatenate([x[0] for x in batch_results], axis=0)

            # Return output
            mean = format_mean(mean)
            if return_std:
                std = np.concatenate([x[1] for x in batch_results], axis=0)
                return mean, format_sigma(std)
            return mean

        # If we reach this point, we have to go through the slow, sequential algorithm
        nn_indices, x_full = self.find_nn_indices_for_train_and_eval(
            kernel_params, x_train, X, condition_on_eval=conditioning_method != 'only-train',
            double_nn=conditioning_method == 'full-double-nn')
        if nn_indices.shape[1] > 1:
            nn_indices[:, 1:] = cp.sort(nn_indices[:, 1:], axis=1)  # To access partial covariance elements less randomly
        num_nn = nn_indices.shape[1]
        nn_indices = nn_indices.get()

        # Allocate output
        mean = np.ones((nq + nt, y_dim)) * np.nan
        mean[:nt] = y_train
        partial_cov = None
        partial_cov_nnz = 0

        if return_std or return_cov:
            std = np.ones(nq) * np.nan

            # Create indices to avoid looping when accessing the partial covariance
            if num_nn > 1:
                pc_row_indexes = np.concatenate(
                    [np.zeros(i, dtype=np.int32) + i for i in range(1, num_nn)])
                pc_col_indexes = np.concatenate(
                    [np.arange(i, dtype=np.int32) for i in range(1, num_nn)])
            else:
                pc_row_indexes, pc_col_indexes = np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        # Create or the partial covariance matrix
        if return_std:
            row_ind = np.repeat(np.arange(nn_indices.shape[0], dtype=np.int32), num_nn - 1)
            col_ind = nn_indices[:, 1:].reshape(-1) - nt
            ind = col_ind >= 0
            row_ind, col_ind = row_ind[ind], col_ind[ind]
            del ind
            if row_ind.shape[0] > 0:
                partial_cov = sp.csr_array((np.ones(row_ind.shape[0]) * np.nan, (row_ind, col_ind)))
                assert partial_cov.has_canonical_format
                partial_cov_nnz = partial_cov.nnz
            del row_ind, col_ind

        # Create the full covariance matrix
        if return_cov:
            partial_cov = np.zeros((nq, nq))

        # Allocate temp variables used in loops
        this_y = np.empty((num_nn - 1, y_dim), dtype=y_train.dtype)
        this_cov = np.empty((num_nn - 1, num_nn - 1))
        diag_ind_left, diag_ind_right = np.diag_indices_from(this_cov)

        # Loop over batches of data in case the entire arrays cannot be all stored in memory
        for i_batch in range(int(np.ceil(nq / self.batch_size))):
            i0 = self.batch_size * i_batch
            i1 = min(self.batch_size * (i_batch + 1), nq)

            # Evaluates the kernel and kernel gradient
            K, _ = self.fill_nn_kernel(
                x_full, cp.array(nn_indices[i0:i1]), kernel_params, theta_mapping, eval_gradient=False)

            # Add jitter to the kernel
            ind = np.where((nn_indices[i0:i1] < nt) & (nn_indices[i0:i1] >= 0))
            K[ind[0], ind[1], ind[1]] += self.alpha
            del ind

            # Calculate the Cholesky decomposition
            L = cp.linalg.cholesky(K[:, 1:, 1:])
            L, K = L.get(), K.get()

            # Fill output
            for i in range(L.shape[0]):
                i_full = i + i0
                this_ind = nn_indices[i_full]
                assert this_ind[0] == nt + i_full
                this_ind = this_ind[1:]
                valid_ind_mask = this_ind > -1
                valid_ind = this_ind[valid_ind_mask]
                this_y[valid_ind_mask] = mean[valid_ind]
                this_y[~valid_ind_mask] = 0

                this_K_xn = K[i, 0, 1:].reshape((1, -1))
                this_K_nn_inv = lambda right: sl.cho_solve((L[i], True), right, overwrite_b=False)

                mean[i_full + nt] = this_K_xn @ this_K_nn_inv(this_y)

                if return_std or return_cov:
                    # Calculate the local covariance
                    nontrain_ind_mask = this_ind >= nt
                    nontrain_ind_pos = np.where(nontrain_ind_mask)[0]
                    nontrain_ind = this_ind[nontrain_ind_mask] - nt

                    this_cov[:, :] = 0
                    n_el = int(nontrain_ind_pos.shape[0] * (nontrain_ind_pos.shape[0] - 1) / 2)
                    if n_el > 0:
                        tmp_row_ind, tmp_col_ind = pc_row_indexes[:n_el], pc_col_indexes[:n_el]
                        this_cov[nontrain_ind_pos[tmp_row_ind], nontrain_ind_pos[tmp_col_ind]] = partial_cov[
                            nontrain_ind[tmp_row_ind], nontrain_ind[tmp_col_ind]]
                        this_cov += this_cov.T
                    this_cov[diag_ind_left[nontrain_ind_mask], diag_ind_right[nontrain_ind_mask]] = std[nontrain_ind]

                    # Calc std
                    std2_raw = max(0, K[i, 0, 0] - (this_K_xn @ this_K_nn_inv(this_K_xn.T))[0, 0])
                    std[i_full] = max(0, std2_raw + (this_K_xn @ this_K_nn_inv(this_cov @ this_K_nn_inv(this_K_xn.T)))[0, 0])

                    # Fill covariance
                    if return_cov:
                        if len(valid_ind) > 0:
                            A = (this_K_xn @ np.linalg.inv(K[i, 1:, 1:]))[:, nontrain_ind_mask]
                            partial_cov[i_full, :i_full] = (A @ partial_cov[nontrain_ind, :i_full]).reshape(-1)
                            partial_cov[:i_full, i_full] = partial_cov[i_full, :i_full]
                        partial_cov[i_full, i_full] = std[i_full]
                    else:
                        # If the full covariance is not needed, we can simply store the covariance between this point and
                        # its nearest neighbours, which is needed later on for the standard deviation of the subsequent
                        # points
                        if len(nontrain_ind) > 0:
                            partial_cov[i_full * np.ones(len(nontrain_ind)), nontrain_ind] = (
                                    this_K_xn @ this_K_nn_inv(this_cov)).reshape(-1)[nontrain_ind_mask]

        # Calculation is done. Wrap up output
        mean = format_mean(mean[nt:])  # Remove training part and format output

        if return_std:
            if partial_cov is not None:
                assert partial_cov.nnz == partial_cov_nnz, "Unexpected error in the partial covariance structure. " \
                                                           "Contact the developer"
            std = np.sqrt(std)
            return mean, format_sigma(std)

        if return_cov:
            if hasattr(self, '_y_train_std'):
                partial_cov = partial_cov[:, :, np.newaxis] * self._y_train_std.reshape((1, 1, -1)) ** 2
                if partial_cov.shape[2] == 1:
                    partial_cov = partial_cov[:, :, 0]
            return mean, partial_cov

        return mean

    @staticmethod
    def get_default_kernel():
        """Returns the default kernel to use when no kernel is specified by the user

        Parameters
        ----------

        Returns
        -------

        kernel: kernels.Kernel
            Default kernel

        """
        return kernels.ConstantKernel(1.0, constant_value_bounds="fixed") * kernels.RBF(
            1.0, length_scale_bounds="fixed")
    
    @staticmethod
    def check_cupy_random_state(random_state):
        """Turn seed into a cp.random.RandomState instance.

        Parameters
        ----------
        seed : None, int or instance of cp.random.RandomState
            If seed is None, return the RandomState singleton used by cp.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.

        Returns
        -------
        :class:`cupy:cupy.random.RandomState`
            The random state object based on `seed` parameter.

        """
        if (random_state is None) or isinstance(random_state, numbers.Integral):
            return cp.random.RandomState(random_state)
        # In this case random_state must be a valid cupy.random.RandomState
        assert isinstance(random_state, cp.random.RandomState), "Random state is not a cupy.random.RandomState instance"
        return random_state

    def find_nearest_neighbours(self, kernel_params, num_nn, ref, query, ref_cut_index=-1):
        """Finds the nearest neighbors in 'ref' for each point in 'query'. It is based on a custom cuda kernel because of the 
        'ref_cut_index' parameter, otherwise could have been easily implemented e.g. on Keops.

        Parameters
        ----------
        kernel_params : dict
            Dictionary containing the parameters of the kernel, as returned by the method `get_kernel_params`

        num_nn : int
            number of nearest  neighbors to use

        ref : cp.ndarray of shape (n_reference_points, n_features)
            Search points for the nearest neighbors search

        query : cp.ndarray of shape (n_query_points, n_features)
            Query points for the nerest neighbors search

        ref_cut_index : int, default -1
            If negative, the nearest neighbours for the j-th point in query are searched for among all points in ref.
            If positive, the nearest neighbours for the j-th point in query are searched for only among points in ref[:j + ref_cut_index].

        Returns
        -------
        nn_indices : np.ndarray of shape (n_query_points, num_nn)
            Nearest neighbour indices. nn_indices[i, j] contains the index of the j-th nearest neighbor in `ref` of the i-th point in
            `query`. If the j-th nearest neighbor for the i-th query points does not exist (e.g. because the search space has less than
            j points, possibly due to the usage of non-negative `ref_cut_index`), then nn_indices[i, j] is set to -1.
        """

        if self.nn_type == 'kernel-space' and 'length_scale' in kernel_params:
            ref = ref / (kernel_params['length_scale'] * kernel_params['rbf_level'])
            query = query / (kernel_params['length_scale'] * kernel_params['rbf_level'])

        # Downcast to float 32 for faster nn search
        if self.allow_downcast_f32 != 'no':
            if ref.dtype != np.float32:
                ref = ref.astype(np.float32)
            if query.dtype != np.float32:
                query = query.astype(np.float32)

        # Start building the kernel header
        data_type = np.promote_types(ref.dtype, query.dtype)
        defines = {} 
        if data_type == np.dtype(np.float64):
            defines['DATA_TYPE'] = "double"
            defines['INFINITY'] = '__longlong_as_double(0x7ff0000000000000ULL)'
            defines['NEG_INFINITY'] = '__longlong_as_double(0xfff0000000000000ULL)'
        elif data_type == np.dtype(np.float32):
            defines['DATA_TYPE'] = "float"
            defines['INFINITY'] = '__int_as_float(0x7f800000)'
            defines['NEG_INFINITY'] = '__int_as_float(0xff800000)'
        elif data_type == np.dtype(np.float16):
            defines['DATA_TYPE'] = "half"
            defines['INFINITY'] = '__short_as_half(0x7c00)'
            defines['NEG_INFINITY'] = '__short_as_half(0xfc00)'
        if ref.itemsize > query.itemsize:
            query = query.astype(ref.dtype)
        if ref.itemsize < query.itemsize:
            ref = ref.astype(query.dtype)

        # Add float16 support
        kernel_headers = []
        if ref.itemsize == 2:
            kernel_headers.append('#include <cuda_fp16.h>')

        # Helper function
        def _get_dtype_for_range(x):
            if x <= np.iinfo(np.int32).max:
                cuda_type = "int"
                cupy_type = cp.intc
            elif x <= np.iinfo(np.int64).max:
                # signed integers somehow performs better than unsigned
                cuda_type = "long long"
                cupy_type = cp.int64
            else:
                assert x <= np.iinfo(np.uint64).max, "Unhandled case of size_t overflow in cuda kernel"
                cuda_type = "size_t"
                cupy_type = cp.intp

            return cuda_type, cupy_type

        # Define types
        max_int_32 = np.iinfo(np.int32).max
        assert ref.shape[1] < max_int_32, "Unhandled case of int overflow in cuda kernel"
        assert num_nn < max_int_32, "Unhandled case of int overflow in cuda kernel"
        defines['INDEX_TYPE'], index_type = _get_dtype_for_range(max(query.shape[0], ref.shape[0]))
        defines['N_QUERY'] = query.shape[0]
        defines['N_REF'] = ref.shape[0]
        defines['NUM_K'] = num_nn
        defines['NUM_FEATURE'] = ref.shape[1]

        # Create common variables
        query_nb = query.shape[0]
        index_gpu = cp.empty(query.shape[0] * num_nn, dtype=index_type).reshape((num_nn, query.shape[0]))
        use_x_transposed = True

        # Number of NNs to keep in cache
        k_for_cache = num_nn

        block_dim_nn = 256
        shared_mem_req_size = k_for_cache * block_dim_nn * (data_type.itemsize + index_type(1).itemsize)
        defines['SIZE_TYPE'], _ = _get_dtype_for_range(max(query.size, ref.size, k_for_cache * block_dim_nn))

        block_size = (block_dim_nn,)
        grid_size = (int(np.ceil(query_nb / block_size[0])),)

        opt_call_kwargs = {}
        opt_call_k_args = []
        cache_dist, cache_index = None, None

        # TODO: get the right static memory size for the device
        max_static_mem_size: int = 49152
        max_static_mem_size_optin: int = 98304
        
        if shared_mem_req_size <= max_static_mem_size:
            cache_memory_type = 'shared_static'
        elif shared_mem_req_size <= max_static_mem_size_optin * 0.9:
            cache_memory_type = 'shared_dynamic'
        else:
            cache_memory_type = 'global'

        if cache_memory_type == 'shared_static':
            defines['USE_STATIC_SHARED_MEM_CACHE'] = 'true'
            dynamic_shared_mem_size = 0
        elif cache_memory_type == 'shared_dynamic':
            defines['USE_DYNAMIC_SHARED_MEM_CACHE'] = 'true'
            dynamic_shared_mem_size = shared_mem_req_size
            opt_call_kwargs = {'shared_mem': dynamic_shared_mem_size}
        elif cache_memory_type == 'global':
            cache_dist = cp.empty(shape=(k_for_cache, query_nb), dtype=data_type)
            cache_index = cp.empty(shape=(k_for_cache, query_nb), dtype=index_type)
            cache_dist[:] = 0
            defines.update({
                'USE_GLOBAL_MEM_CACHE': 'true',
                'CACHE_DIST_PITCH': '*' + str(cache_dist.shape[1]), 
                'CACHE_INDEX_PITCH': '*' + str(cache_index.shape[1])
            })
            dynamic_shared_mem_size = 0
            opt_call_k_args = [cache_dist, cache_index]
        else:
            raise RuntimeError("Unexpcted cache_memory_type: " + str(cache_memory_type))

        if use_x_transposed:
            defines['REF_PITCH_FEATURE'] = '* ' + str(ref.shape[0])  # ideally query should be in a pitched memory layout
            defines['REF_PITCH_POINT'] = ''
        else:
            defines['REF_PITCH_FEATURE'] = ''
            defines['REF_PITCH_POINT'] = '* ' + str(ref.shape[0])  # ideally query should be in a pitched memory layout

        # Initialize veriables
        if use_x_transposed:
            ref = ref.T
        ref = cp.array(cp.ascontiguousarray(ref) if type(ref) is cp.ndarray else np.ascontiguousarray(ref))
        query = query.T
        query = cp.array(cp.ascontiguousarray(query) if type(query) is cp.ndarray else np.ascontiguousarray(query))
        query_pitch = query.shape[1]  # ideally query should be in a pitched memory layout

        # Determine max search point
        restrict_nn_index = None if ref_cut_index < 0 else ref_cut_index
        if restrict_nn_index is None:
            defines['MAX_SEARCH_POINT'] = defines['N_REF']
        else:
            defines['MAX_SEARCH_POINT'] = '(i_query + ' + str(restrict_nn_index) + ')'
            index_gpu[:, :max(0, num_nn - restrict_nn_index)] = -1

        # Build the kernel with the desired options
        defines['BLOCK_DIM'] = block_dim_nn
        defines['QUERY_PITCH'] = query_pitch
        defines['NUM_K_FOR_CACHE'] = k_for_cache
        defines['BLOCK_DIM'] = block_dim_nn
        defines['INDEX_PITCH'] = index_gpu.shape[1]

        kernel_headers = kernel_headers + ["#define " + k + " " + str(v) for k, v in defines.items()] + ["", ""]
        kernel_headers = "\n".join(kernel_headers)
        krn = cp.RawKernel(kernel_headers + _calc_nn_kernel_text, 'calc_nn')

        # Run
        if dynamic_shared_mem_size > 0:
            krn.max_dynamic_shared_size_bytes = dynamic_shared_mem_size
        cp.cuda.Device(index_gpu.device).synchronize()
        krn(
            grid_size, block_size, 
            (ref, query, index_gpu, *opt_call_k_args), **opt_call_kwargs)
        cp.cuda.Device(index_gpu.device).synchronize()
        return cp.transpose(index_gpu)
    
    @staticmethod
    def _build_ytrain_given_nn_indices(y_train, nn_indices):
        """Calculates the array containing the observed y values for each nearest neighbor.

        Parameters
        ----------

        y_train : np.ndarray or cp.ndarray of shape (n_train,)
            Observed y values in the training dataset.

        nn_indices : cp.ndarray of shape (n_query, n_nn)
            Array that at position [i, j] stores the index of the j-th nearest neighbor for the i-th query point. nn_indices[i, j]
            must be set to -1 when the j-th nearest neighbor for the i-th query point is not defined.

        Returns
        -------

        y_train_nn : cp.ndarray of the same shape (n_query, n_nn)
            Array that at position [i, j] stores the observed y_train on the j-th nearest neighbor for the i-th query point. If the j-th
            nearest neighbor index in nn_indices is -1, then y_train_nn[i, j] is set to zero.

        """

        if type(y_train) is not cp.ndarray:
            y_train = cp.array(y_train)
        y_train_nn = np.empty((*nn_indices.shape, y_train.shape[-1]), y_train.dtype)
        usable = (nn_indices != -1).get()
        y_train_nn[usable, :] = y_train[nn_indices[usable], :].get()
        y_train_nn[~usable, :] = 0

        return y_train_nn

    def fill_nn_kernel(self, x, nn_indices, kernel_params, kernel_theta_mapping, eval_gradient=False):
        """Calculates the kernel based on nearest neighbors given the nearest neighbors indices.

        Parameters
        ----------
        x : np.ndarray or cp.ndarray of shape (n_points, n_features)
            Input dataset

        nn_indices : np.ndarray or cp.ndarray of shape (n_query, n_nn)
            Array that at position [i, j] stores the index of the j-th nearest neighbor for the i-th query point. nn_indices[i, j]
            must be set to -1 when the j-th nearest neighbor for the i-th query point is not defined.

        kernel_params : dict
            Dictionary containing the parameters of the kernel, as returned by the method `get_kernel_params`

        kernel_theta_mapping : dict
            Dictionary containing the mapping from kernel parameter name to kernel parameter index, as returned by the method `get_kernel_theta_mapping`

        eval_gradient : bool, default False
            If True the kernel gradient is also evaluated

        Returns
        -------
        K : np.ndarray of the same shape (n_query, n_nn, n_nn)
            K[i, j, k] is the kernel evaluated between the j-th and k-th nearest neighbors of the i-th query point.

        K_gradient : np.ndarray of the same shape (n_query, n_nn, n_nn, n_theta)
            K[i, j, k, :] is the kernel gradient evaluated between the j-th and k-th nearest neighbors of the i-th query point. Returned
            only if 'eval_gradient' is set to True.

        """

        if (type(x) is np.float64) and (self.allow_downcast_f32 == 'yes'):
            x = x.astype('float32')
        if type(x) is not cp.ndarray:
            x = cp.array(x)
        if type(nn_indices) is not cp.ndarray:
            nn_indices = cp.array(nn_indices.T)
        else:
            nn_indices = nn_indices.T  # For faster cuda access

        # Prepare data to be accessed by the kernel
        if not x.flags['C_CONTIGUOUS']:
            x = cp.ascontiguousarray(x)
        assert x.flags['C_CONTIGUOUS']
        if not nn_indices.flags['C_CONTIGUOUS']:
            nn_indices = cp.ascontiguousarray(nn_indices)
        assert nn_indices.flags['C_CONTIGUOUS']
        assert nn_indices.dtype in [np.int32, np.int64], "Unsupported dtype for nn_indices: " + str(nn_indices.dtype)

        # Allocate output
        n_theta = sum(v.size if type(v) is np.ndarray else 1 for k, v in kernel_theta_mapping.items())
        nn = nn_indices.shape[0]
        nn_kernel = cp.zeros((nn, nn, nn_indices.shape[1]), dtype=x.dtype)
        nn_kernel_grad = cp.zeros((nn, nn, n_theta, nn_indices.shape[1]), dtype=x.dtype) if eval_gradient else cp.zeros((0, 0, 0, 0))
        rbf_length_is_vec = not isinstance(kernel_params.get('length_scale', 1), numbers.Number)
        rbf_length_scales = cp.array(kernel_params['length_scale'].reshape(-1), dtype=x.dtype) if rbf_length_is_vec \
            else cp.array(kernel_params.get('length_scale', 1), dtype=x.dtype)

        # Check dtypes
        if x.dtype.type == np.float64:
            data_type = 'double'
            exp_fun = 'exp'
        elif x.dtype.type == np.float32:
            data_type = 'float'
            exp_fun = 'expf'
        elif x.dtype.type == np.float16:
            data_type = 'single'
            exp_fun = 'h2exp'
        else:
            raise NotImplementedError(str(x.dtype))

        min_i_type = np.min_scalar_type(max(x.shape[0], x.shape[1], nn_indices.shape[0], nn_indices.shape[1]))
        if np.can_cast(min_i_type, 'int32'):
            index_type = 'int'
        elif np.can_cast(min_i_type, 'int64'):
            index_type = 'long long'
        else:
            raise NotImplementedError(str(min_i_type))

        min_i_type = np.min_scalar_type(max(nn_kernel.size, nn_kernel_grad.size))
        if np.can_cast(min_i_type, 'int32'):
            big_index_type = 'int'
        elif np.can_cast(min_i_type, 'int64'):
            big_index_type = 'long long'
        elif np.can_cast(min_i_type, 'uint64'):
            big_index_type = 'size_t'
        else:
            raise NotImplementedError(str(min_i_type))

        # Build kernel text
        block_dim_nn = 64
        defines = {
            'BLOCK_DIM': block_dim_nn,
            'EXP': exp_fun,

            'NUM_REF_POINTS': nn_indices.shape[1],
            'NUM_NN': nn_indices.shape[0],
            'NUM_FEATURES': x.shape[1],
            'NUM_THETA': n_theta,

            'PITCH_NN_POINTS': nn_indices.shape[1],
            'PITCH_DATA_POINTS': x.shape[1],
            'PITCH_NN_KERNEL': nn_kernel.shape[-1],
            'PITCH_NN_KERNEL_GRAD': nn_kernel_grad.shape[-1] * nn_kernel_grad.shape[-2],
            'THETA_PITCH': nn_kernel_grad.shape[-1],

            'DATA_TYPE': data_type,
            'INDEX_TYPE': index_type,
            'NN_IND_DTYPE': 'long long' if nn_indices.dtype == np.int64 else 'int',
            'BIG_INDEX_TYPE': big_index_type,

            'WHITE_NOISE_INDEX': kernel_theta_mapping.get('noise_level', -1) if eval_gradient else -1,
            'CONSTANT_LEVEL_INDEX': kernel_theta_mapping.get('constant_value', -1) if eval_gradient else -1,
            'RBF_LEVEL_INDEX': kernel_theta_mapping.get('rbf_level', -1) if eval_gradient else -1,
            'RBF_SCALE_INDEX': kernel_theta_mapping.get('length_scale', -1) if eval_gradient and not rbf_length_is_vec else -1,
            'RBF_SCALE_VEC_INDEX_0': kernel_theta_mapping.get('length_scale', [-1])[0] if eval_gradient and rbf_length_is_vec else -1,
            'RBF_SCALE_VEC_INDEX_1': kernel_theta_mapping.get('length_scale', [-1])[-1] + 1 if eval_gradient and rbf_length_is_vec else -1,
        }
        if data_type == np.float16:
            defines['USE_F16'] = 1
        if rbf_length_is_vec:
            defines['IS_VEC_RBF_SCALE'] = 1
        lines = ["#define " + k + " " + str(v) for k, v in defines.items()] + ["", ""]
        kernel_text = "\n".join(lines) + _fill_nn_kernel_text

        # Execute kernel
        krn = cp.RawKernel(kernel_text, 'fill_nn_kernel')
        block_size = (block_dim_nn,)
        grid_size = (int(np.ceil(nn_indices.shape[1] / block_size[0])),)

        cp.cuda.Device(nn_kernel.device).synchronize()
        krn(grid_size, block_size, (
            x, nn_indices, rbf_length_scales, nn_kernel, nn_kernel_grad,
            x.dtype.type(kernel_params.get('constant_value', 0)),
            x.dtype.type(kernel_params.get('noise_level', 0)),
            x.dtype.type(kernel_params.get('rbf_level', 1 if 'length_scale' in kernel_params else 0))))
        cp.cuda.Device(nn_kernel.device).synchronize()

        # Post-processing
        nn_kernel = cp.moveaxis(nn_kernel, -1, 0)  # Follow sickitlearn convention
        nn_kernel_grad = cp.moveaxis(nn_kernel_grad, -1, 0)  # Follow sickitlearn convention

        return nn_kernel, nn_kernel_grad

    def find_nn_indices_for_train_and_eval(self, kernel_params, x_train, x_query, condition_on_eval, double_nn):
        """Calculates the array containing the observed y values for each nearest neighbor.

        Parameters
        ----------

        kernel_params : dict
            Dictionary containing the parameters of the kernel, as returned by the method `get_kernel_params`

        x_train : np.ndarray of shape (n_train, n_features)
            Array containing the training points

        x_query : np.ndarray of shape (n_query, n_features)
            Array containing the points for which nearest neighbors are searched

        condition_on_eval : bool
            If True, nearest neighbors for the i-th point in `x_query' are searched in x_train and x_query[:i, :]. If False, nearest
            neighbors are searched only in x_train

        double_nn : bool
            If True, the number of nearest neighbors (self._num_nn_) searched for every point in x_query is doubled. When True, ror the
            i-th query point, self._num_nn_ neighbors are searched in x_train, and other self._num_nn_ are searched in x_query[:i, :].
            Used only when condition_on_eval is True.

        Returns
        -------

        nn_indices : cp.ndarray of shape (n_query, n_nn + 1) or (n_query, n_nn*2 + 1)
            Array that at position [i, j], when j > 0, stores the index of the j-th nearest neighbor for the i-th query point.
            nn_indices[i, 0] stores the index of the i-th query point. Indices are meant to be used to lookup points inside the `x_full`
            array, which is returned by this function as second argument. When the j-th nearest neighbor for the i-th query point does not
            exist, then nn_indices[i, j] is set to -1.

        x_full : np.ndarray of shape (n, n_features)
            Array that can be used to lookup points given the indices stored in nn_indices. It's either equal to `x_train` or the
            concatenation of `x_train` and `x_query` depending on the input variables.

        """

        def get_x_full():
            return np.concatenate([x_train, x_query])

        x_full = None
        if condition_on_eval:
            if double_nn:
                if x_train.shape[0] > 0:
                    nn_ind_train = self.find_nearest_neighbours(kernel_params, min(self.num_nn, x_train.shape[0]), x_train, x_query)
                else:
                    nn_ind_train = np.empty((0, 0)), np.empty((0, 0), dtype=np.int32)
                nn_ind_nontrain = self.find_nearest_neighbours(
                    kernel_params, min(self.num_nn, x_query.shape[0] - 1), x_query, x_query, ref_cut_index=0)
                nn_ind_nontrain[nn_ind_nontrain != -1] += x_train.shape[0]
                if x_train.shape[0] > 0:
                    # Need to combine nn_ind_train and nn_ind_nontrain
                    nn_indices = cp.empty((nn_ind_train.shape[0], nn_ind_train.shape[1] + nn_ind_nontrain.shape[1]),
                                          dtype=nn_ind_train.dtype)
                    arange = cp.tile(cp.arange(nn_indices.shape[1], dtype=cp.int32), (nn_indices.shape[0], 1))

                    # First insert indices from indices_0
                    is_valid = nn_ind_train != -1
                    n_to_add_0 = cp.sum(is_valid, axis=1).reshape((-1, 1))
                    nn_indices[arange < n_to_add_0] = nn_ind_train[is_valid]

                    # Then insert indices from indices_1
                    is_valid = nn_ind_nontrain != -1
                    n_to_add_1 = n_to_add_0 + cp.sum(is_valid, axis=1).reshape((-1, 1))
                    nn_indices[(arange >= n_to_add_0) & (arange < n_to_add_1)] = nn_ind_nontrain[is_valid]

                    # Set the rest to the null value
                    nn_indices[arange >= n_to_add_1] = -1
                else:
                    nn_indices = nn_ind_nontrain
                del nn_ind_train, nn_ind_nontrain
            else:
                x_full = get_x_full()
                nn_indices = self.find_nearest_neighbours(
                    kernel_params, min(self.num_nn, x_full.shape[0]), x_full, x_query, ref_cut_index=x_train.shape[0])
        else:
            num_nn = min(x_train.shape[0], self.num_nn)
            if num_nn > 0:
                nn_indices = self.find_nearest_neighbours(kernel_params, num_nn, x_train, x_query)
            else:
                nn_indices = cp.empty((x_query.shape[0], 0), dtype=np.int32)

        nn_indices = cp.concatenate([
            cp.arange(x_train.shape[0], x_train.shape[0] + nn_indices.shape[0]).reshape((-1, 1)), nn_indices], axis=1)
        x_full = get_x_full() if x_full is None else x_full
        return nn_indices, x_full

    @staticmethod
    def _free_cupy_memory():
        """Helper functions that frees some allocated but unused memory by cupy

        Parameters
        ----------

        
        Returns
        -------


        """
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    #TODO: replace kernel parameter extraction with cuda-based kernels
    @staticmethod
    def get_kernel_params(kernel):
        """Helper functions that extract the kernel parameters and places them in a dict. Currently, only kernels of the type
        WhiteKernel + ConstantKernel  * RBF are supported. This function is intentionally not well documented since it is going
        to be replaced by cuda-based kernel classes.

        Parameters
        ----------
        kernel : kernel.Kernel
            Input kernel whose parameters will be extracted and returned as output
        
        Returns
        -------
        kernel_parameters : dict
            Dictionary that maps each kernel parameter to its value

        """
        if issubclass(type(kernel), kernels.WhiteKernel):
            kernel: kernels.WhiteKernel
            return {'noise_level': kernel.noise_level}

        if issubclass(type(kernel), kernels.ConstantKernel):
            kernel: kernels.ConstantKernel
            return {'constant_value': kernel.constant_value}

        if issubclass(type(kernel), kernels.RBF):
            kernel: kernels.RBF
            return {'length_scale': kernel.length_scale if isinstance(kernel.length_scale, numbers.Number)
            else np.array(kernel.length_scale).reshape((1, -1))}

        if issubclass(type(kernel), kernels.Sum):
            kernel: kernels.Sum
            p1 = CUNNGPR.get_kernel_params(kernel.k1)
            p2 = CUNNGPR.get_kernel_params(kernel.k2)
            # Sum is supported only with a white kernel
            is_p1_rbf = len(set(p1.keys()).difference({'noise_level', 'constant_value'})) > 0
            is_p2_rbf = len(set(p2.keys()).difference({'noise_level', 'constant_value'})) > 0
            if is_p1_rbf and is_p2_rbf:
                raise RuntimeError("Sum of kernels is not supported between RBF kernels. Got sum between "
                                   + str(p1) + " and " + str(p2))
            if is_p1_rbf:
                p1, p2 = p2, p1
            # p1 is the non rbf kernel
            if ('noise_level' in p1) or ('noise_level' in p2):
                p2['noise_level'] = p1.get('noise_level', 0) + p2.get('noise_level', 0)
            if ('constant_value' in p1) or ('constant_value' in p2):
                p2['constant_value'] = p1.get('constant_value', 0) + p2.get('constant_value', 0)
            return p2

        if issubclass(type(kernel), kernels.Product):
            kernel: kernels.Product
            p1 = CUNNGPR.get_kernel_params(kernel.k1)
            p2 = CUNNGPR.get_kernel_params(kernel.k2)
            # Product is supported only with a ConstantKernel
            is_p1_const = len(set(p1.keys()).difference({'constant_value'})) == 0
            is_p2_const = len(set(p2.keys()).difference({'constant_value'})) == 0
            if not is_p1_const and not is_p2_const:
                raise RuntimeError("Product of kernels is supported only with a ConstantKernel. Got product between "
                                   + str(p1) + " and " + str(p2))
            if is_p2_const:
                p1, p2 = p2, p1
            # p1 is the constant kernel
            if 'noise_level' in p2:
                p2['noise_level'] = p1['constant_value'] * p2['noise_level']
            if 'constant_value' in p2:
                p2['constant_value'] = p1['constant_value'] * p2['constant_value']
            if ('rbf_level' in p2) or ('length_scale' in p2):
                p2['rbf_level'] = p1['constant_value'] * p2.get('rbf_level', 1)
            return p2

        raise RuntimeError("Unsupported kernel type: " + str(type(kernel)))

    #TODO: replace kernel parameter extraction with cuda-based kernels
    @staticmethod
    def get_kernel_theta_mapping(kernel):
        """Helper functions that extract the kernel parameters and places them in a dict. Currently, only kernels of the type
        WhiteKernel + ConstantKernel  * RBF are supported. This function is intentionally not well documented since it is going
        to be replaced by cuda-based kernel classes.

        Parameters
        ----------
        kernel : kernel.Kernel
            Input kernel whose parameters will be extracted and returned as output
        
        Returns
        -------
        kernel_parameters : dict
            Dictionary that maps each kernel parameter to its value

        """

        n_theta = len(kernel.theta)
        epsi = 1e-15
        kernel = kernel.clone_with_theta(np.arange(n_theta))
        expected = np.exp(kernel.theta)
        new_kernel_params = CUNNGPR.get_kernel_params(kernel)

        def find_index_for_param(param_name):
            param = new_kernel_params[param_name]
            if (param_name != 'length_scale') or (isinstance(param, numbers.Number)):
                found = (param < expected * (1 + epsi)) & (param > expected * (1 - epsi))
                n_found = np.sum(found)
                if n_found > 1:
                    raise RuntimeError("Cannot find theta mapping for parameter " + param_name)
                if n_found == 0:
                    return None
                return np.where(found)[0][0]

            params = new_kernel_params[param_name].reshape(-1)
            found = np.zeros(len(expected), dtype=bool)
            for param in params:
                tmp = (param < expected * (1 + epsi)) & (param > expected * (1 - epsi))
                if np.sum(tmp) == 1:
                    found[np.where(tmp)[0][0]] = 1
            n_found = np.sum(found)
            if n_found > params.size:
                raise RuntimeError("Cannot find theta mapping for parameter " + param_name)
            if n_found == 0:
                return None
            ind = np.where(found)[0]
            assert (len(ind) == 1) or (set(np.unique(np.diff(ind))) == {1}), "Unsupported case of non-adjacent parameters for length scale"
            return ind

        theta_mapping = {param: find_index_for_param(param) for param in new_kernel_params}
        theta_mapping = {k: v for k, v in theta_mapping.items() if v is not None}
        num_params_found = sum(len(v) if type(v) is np.ndarray else 1 for k, v in theta_mapping.items())
        if num_params_found < len(kernel.theta):
            raise RuntimeError("Cannot find the mapping for all kernel thetas")
        assert num_params_found == len(kernel.theta), "Unexpected case when mapping kernel thetas to parameters"

        return theta_mapping


def cusolver_potrs(L: cp.ndarray, b: cp.ndarray, lower: bool, check_finite: bool = False):
    """ Implements lapack XPOTRS through cusolver.potrs. Solves linear system A * x = b given the cholesky decomposition of A. 

    Parameters
    ----------
    L : cp.ndarray of shape (...,n,n)
        Cholesky decomposition of one or many square matrices of shape n,n. The decomposition must be stored along the last
        two dimensions of the array L, with the other dimensions representing batch-dimensions.

    b : cp.ndarray of shape (...,n,m)
        Right hand side of the matrix equatons to solve. The RHS must be stored along the last two dimensions of the array b, 
        with the other dimensions representing batch-dimensions. The last dimension m represents the number of right hand side.
        Dimensions except the last two must match the dimensions in the input parameter L

    lower : bool
        If True, each cholesky decompoistion is stored in the lower triangular part of L, otherwise in the upper triangulat.
    
    check_finite : bool
        If True, input arrays are checked for finitness and an excpetion is raised if they are not finite.

    Returns
    -------
    X : cp.ndarray of shape (...,n,m)
        Solution to the linear system A * x = b

    """

    # Check if batched should be used
    if len(L.shape) > 2:
        return cusolver_potrs_batched(L, b, lower=lower, check_finite=check_finite)

    # Check input arguments
    one_dim = False
    if len(b.shape) == 1:
        b = b.reshape((-1, 1))
        one_dim = True
    else:
        assert len(b.shape) == 2, "B must be either a column vector or a matrix"
    assert len(L.shape) == 2, "L must be a square matrix"
    assert L.shape[0] == L.shape[1], "L must be a square matrix"
    assert b.shape[0] == L.shape[0], "L and b shape mismatch"
    dtype = np.common_type(L, b)

    # Check for null input
    if L.size == 0:
        return cp.empty(b.squeeze().shape, dtype=dtype)

    # Check for non cuda array
    assert type(L) is cp.ndarray, "Input must be a cupy array"
    assert type(b) is cp.ndarray, "Input must be a cupy array"

    # Check finiteness
    if check_finite:
        assert cp.all(cp.isfinite(L)).get(), "Non real numbers in L"
        assert cp.all(cp.isfinite(b)).get(), "Non real numbers in b"

    # Check memory order and type
    if (L.dtype != dtype) or ((not L.flags['F_CONTIGUOUS']) and (not L.flags['C_CONTIGUOUS'])):
        L = L.astype(dtype, order='F')
    if L.flags['C_CONTIGUOUS']:
        lower = not lower
    if (b.dtype != dtype) or (not b.flags['F_CONTIGUOUS']):
        b = b.astype(dtype, order='F')

    # Variables for potrs
    ldL = L.shape[0]
    ldB = b.shape[0]
    handle = device.get_cusolver_handle()
    dev_info = cp.empty(1, dtype=np.int32)

    # Take correct dtype
    if dtype == np.float32:
        potrs = cusolver.spotrs
    elif dtype == np.float64:
        potrs = cusolver.dpotrs
    elif dtype == np.complex64:
        potrs = cusolver.cpotrs
    else:  # dtype == np.complex128:
        potrs = cusolver.zpotrs

    with cpx.errstate(linalg='raise'):
        potrs(
            handle, cublas.CUBLAS_FILL_MODE_LOWER if lower else cublas.CUBLAS_FILL_MODE_UPPER, L.shape[0], b.shape[1],
            L.data.ptr, ldL,
            b.data.ptr, ldB, dev_info.data.ptr)

    dev_info = dev_info.get()[0]
    if dev_info != 0:
        raise cp.linalg.LinAlgError(
            'Error reported by potrs in cuSOLVER. devInfo = {}. Please refer'
            ' to the cuSOLVER documentation.'.format(dev_info))
    if one_dim:
        b = b.reshape(-1)

    return b


def cusolver_potrs_batched(L: cp.ndarray, b: cp.ndarray, lower: bool, check_finite: bool = False):
    """ Implements lapack XPOTRS through cusolver.potrsBatched. Solves a batch of linear systems A * x = b given the cholesky 
    decompositions of A. 

    Parameters
    ----------
    L : cp.ndarray of shape (...,n,n)
        Cholesky decomposition many square matrices of shape n,n. The decomposition must be stored along the last
        two dimensions of the array L, with the other dimensions representing batch-dimensions.

    b : cp.ndarray of shape (...,n,m)
        Right hand side of the matrix equatons to solve. The RHS must be stored along the last two dimensions of the array b, 
        with the other dimensions representing batch-dimensions. The last dimension m represents the number of right hand side.
        Dimensions except the last two must match the dimensions in the input parameter L

    lower : bool
        If True, each cholesky decompoistion is stored in the lower triangular part of L, otherwise in the upper triangulat.
    
    check_finite : bool
        If True, input arrays are checked for finitness and an excpetion is raised if they are not finite.

    Returns
    -------
    X : cp.ndarray of shape (...,n,m)
        Solution to the linear system A * x = b

    """

    # Check input arguments
    assert len(L.shape) > 2, "L must be at least a 3-d array"
    assert L.shape[-1] == L.shape[-2], "Last two dimensions of L must be the same (square matrices)"
    assert len(b.shape) >= 2, "b must be at least a 2-d array"
    remove_dim = False
    if len(b.shape) == len(L.shape) - 1:
        b = b[:, np.newaxis]
        remove_dim = True
    shape_factor = L.shape[:-2]
    assert b.shape[:-2] == L.shape[:-2], "L and b first dimensions mismatch"
    assert b.shape[-2] == L.shape[-1], "length of arrays in b does not match size of L"
    assert b.shape[:-2] == shape_factor, "number of elements in b does not match L"

    # Check for non cuda array
    assert type(L) is cp.ndarray, "Input must be a cupy array"
    assert type(b) is cp.ndarray, "Input must be a cupy array"

    # Check dtype and memory alignment
    dtype = np.common_type(L, b)
    L = L.reshape((-1, *L.shape[-2:]), order='C')
    b = b.reshape((-1, *b.shape[-2:]), order='C')
    if L.dtype != dtype:
        L = L.astype(dtype)
    if not L.flags['C_CONTIGUOUS']:
        L = cp.ascontiguousarray(L)
    if b.dtype != dtype:
        b = b.astype(dtype)
    if not b.flags['C_CONTIGUOUS']:
        b = cp.ascontiguousarray(b)
    assert L.flags['C_CONTIGUOUS'] and b.flags['C_CONTIGUOUS']
    lower = not lower  # Cuda by default works with F-contiguous blocks, but we have C-contiguous

    # Check for null input
    if L.size == 0:
        return cp.empty(b.shape, dtype=dtype)

    # Check finiteness
    if check_finite:
        assert cp.all(cp.isfinite(L)).get(), "Non real numbers in L"
        assert cp.all(cp.isfinite(b)).get(), "Non real numbers in b"

    # Take correct dtype
    if dtype == np.float32:
        potrsBatched = cusolver.spotrsBatched
    elif dtype == np.float64:
        potrsBatched = cusolver.dpotrsBatched
    elif dtype == np.complex64:
        potrsBatched = cusolver.cpotrsBatched
    elif dtype == np.complex128:
        potrsBatched = cusolver.zpotrsBatched
    else:
        raise NotImplementedError("Unsupported dtype: " + str(dtype))

    # Variables for potrs batched
    batch_size = b.shape[0]
    handle = device.get_cusolver_handle()
    dev_info = cp.empty(1, dtype=cp.int32)
    L_ptrs = cp.arange(batch_size, dtype=cp.uintp) * (
                L.shape[-1] * L.shape[-2] * L.itemsize) + L.data.ptr  # We can do it since it's contiguous
    nrhs = b.shape[-1]
    if nrhs == 1:
        b_tmp = b[..., 0]
    else:
        b_tmp = cp.empty(b.shape[:-1], dtype=b.dtype, order='C')
    b_ptrs = cp.arange(batch_size, dtype=cp.uintp) * b_tmp.shape[-1] * b_tmp.itemsize + b_tmp.data.ptr

    # potrs_batched supports only nrhs=1, so we have to loop
    for i in range(b.shape[-1]):

        if nrhs > 1:  # Copy results back to the original array
            b_tmp[...] = b[..., i]

        with cpx.errstate(linalg='raise'):
            potrsBatched(
                handle,
                cublas.CUBLAS_FILL_MODE_LOWER if lower else cublas.CUBLAS_FILL_MODE_UPPER,
                L.shape[-1],  # n
                1,  # nrhs
                L_ptrs.data.ptr,  # A
                L.shape[1],  # lda
                b_ptrs.data.ptr,  # Barray
                b.shape[1],  # ldb
                dev_info.data.ptr,  # info
                batch_size  # batchSize
            )

        dev_info_value = dev_info.get()[0]
        if dev_info_value != 0:
            raise cp.linalg.LinAlgError(
                'Error reported by potrs in cuSOLVER. devInfo = {}. Please refer'
                ' to the cuSOLVER documentation.'.format(dev_info_value))

        if nrhs > 1:  # Copy results back to the original array
            b[..., i] = b_tmp

    b = b.reshape((*shape_factor, *b.shape[-2:]))
    if remove_dim:
        b = b[..., 0]

    return b


# Cuda kernel text to calculate nearest neighbors
_calc_nn_kernel_text = """
extern "C"
__global__ void calc_nn(
    DATA_TYPE const * ref,
    DATA_TYPE const * query,
    INDEX_TYPE *  k_indexes
#ifdef USE_GLOBAL_MEM_CACHE
    , DATA_TYPE * cache_distances,
    INDEX_TYPE * cache_indexes
#endif
)
{
    // Thread index
    INDEX_TYPE i_query = blockIdx.x * BLOCK_DIM + threadIdx.x;

    if (i_query >= N_QUERY)
        return;

#ifdef USE_STATIC_SHARED_MEM_CACHE
    __shared__ DATA_TYPE cache_distances[BLOCK_DIM][NUM_K_FOR_CACHE];
    __shared__ INDEX_TYPE cache_indexes[BLOCK_DIM][NUM_K_FOR_CACHE];
    DATA_TYPE * this_distances = cache_distances[threadIdx.x];
    INDEX_TYPE * this_indexes = cache_indexes[threadIdx.x];
#define CACHE_DIST_PITCH
#define CACHE_INDEX_PITCH

#elif defined(USE_DYNAMIC_SHARED_MEM_CACHE)
    extern __shared__ DATA_TYPE cache_distances[];
    INDEX_TYPE * cache_indexes = (INDEX_TYPE *) (cache_distances + BLOCK_DIM * NUM_K_FOR_CACHE);
    
#ifdef DYNAMIC_SHARED_MEM_ALT_ORDER
    DATA_TYPE * this_distances = cache_distances + threadIdx.x;
    INDEX_TYPE * this_indexes = cache_indexes + threadIdx.x;
#define CACHE_DIST_PITCH *BLOCK_DIM
#define CACHE_INDEX_PITCH *BLOCK_DIM
#else
    DATA_TYPE * this_distances = cache_distances + threadIdx.x * NUM_K_FOR_CACHE;
    INDEX_TYPE * this_indexes = cache_indexes + threadIdx.x * NUM_K_FOR_CACHE;
#define CACHE_DIST_PITCH
#define CACHE_INDEX_PITCH
#endif

#elif defined(USE_GLOBAL_MEM_CACHE)
    DATA_TYPE * this_distances = cache_distances + i_query;
    INDEX_TYPE * this_indexes = cache_indexes + i_query;
#endif

    DATA_TYPE const * query_point = query + i_query;

    for (SIZE_TYPE j=0; j<NUM_K; j++) this_indexes[j CACHE_INDEX_PITCH ] = -1;
    for (SIZE_TYPE j=0; j<NUM_K; j++) this_distances[j CACHE_DIST_PITCH ] = INFINITY;
    DATA_TYPE max_dist = INFINITY;

    for (SIZE_TYPE i_ref=0; i_ref<MAX_SEARCH_POINT; i_ref++) {
        DATA_TYPE distance = 0;
        DATA_TYPE const * ref_point = ref + i_ref REF_PITCH_POINT ;
        for (SIZE_TYPE f=0; f<NUM_FEATURE; f++) {
            DATA_TYPE tmp = ref_point[f REF_PITCH_FEATURE ] - query_point[f * QUERY_PITCH];
            distance += tmp * tmp;
        }

        // If this is within the k nearest, update it
        if (distance < max_dist) {
            SIZE_TYPE ii = NUM_K - 1;
            while (ii > 0) {
                if (this_distances[(ii - 1) CACHE_DIST_PITCH ] <= distance)
                    break;
                this_distances[ii CACHE_DIST_PITCH ] = this_distances[(ii - 1) CACHE_DIST_PITCH ];
                this_indexes[ii CACHE_INDEX_PITCH ] = this_indexes[(ii - 1) CACHE_INDEX_PITCH ];
                ii -= 1;
            }
            this_distances[ii CACHE_DIST_PITCH ] = distance;
            this_indexes[ii CACHE_INDEX_PITCH ] = (INDEX_TYPE) i_ref;
            max_dist = this_distances[(NUM_K - 1) CACHE_DIST_PITCH ];
        }
    }

    for (SIZE_TYPE i=0; i<min(NUM_K, MAX_SEARCH_POINT); i++) {
        k_indexes[i_query + i * INDEX_PITCH] = this_indexes[i CACHE_INDEX_PITCH ];
    }
}

"""

# Cuda kernel text to calculate the GP kernel and its gradient
_fill_nn_kernel_text = """
#ifdef USE_F16
#include cuda_fp16.h
#endif

extern "C"
__global__ void fill_nn_kernel(
    DATA_TYPE const * data_points,
    NN_IND_DTYPE const * nn_points,
    DATA_TYPE const * rbf_length_scales,
    DATA_TYPE * nn_kernel,
    DATA_TYPE * nn_kernel_grad,
    DATA_TYPE const const_value,
    DATA_TYPE const white_value,
    DATA_TYPE const rbf_level
)
{
    INDEX_TYPE i_ref_point;
    // Thread index
    {
        size_t i_ref_size_t = blockIdx.x * BLOCK_DIM + threadIdx.x;
        if (i_ref_size_t >= NUM_REF_POINTS)
            return;
        i_ref_point = (INDEX_TYPE) i_ref_size_t;
    }
    NN_IND_DTYPE const * this_nn_points = nn_points + i_ref_point;
    DATA_TYPE * this_kernel = nn_kernel + i_ref_point;
    DATA_TYPE * this_kernel_grad = nn_kernel_grad + i_ref_point;
    DATA_TYPE tmp_dist[NUM_FEATURES], exp_value, kernel_value;

    for (INDEX_TYPE i_nn=0; i_nn<NUM_NN; i_nn++) {
        NN_IND_DTYPE const nn_i_index = *(this_nn_points + i_nn * PITCH_NN_POINTS);
        DATA_TYPE const * i_data_point = data_points + nn_i_index * PITCH_DATA_POINTS;
        for (INDEX_TYPE j_nn=0; j_nn<=i_nn; j_nn++) {
            NN_IND_DTYPE const nn_j_index = *(this_nn_points + j_nn * PITCH_NN_POINTS);
            DATA_TYPE const * j_data_point = data_points + nn_j_index * PITCH_DATA_POINTS;
            BIG_INDEX_TYPE ij_index = i_nn + j_nn * NUM_NN;
            BIG_INDEX_TYPE ji_index = j_nn + i_nn * NUM_NN;

            if ((nn_i_index == -1) || (nn_j_index == -1)) {
                kernel_value = (i_nn == j_nn) ? 1 : 0;
                if (WHITE_NOISE_INDEX > -1) {
                    this_kernel_grad[ij_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * WHITE_NOISE_INDEX] = 0;
                    this_kernel_grad[ji_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * WHITE_NOISE_INDEX] = 0;
                }
                if (CONSTANT_LEVEL_INDEX > -1) {
                    this_kernel_grad[ij_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * CONSTANT_LEVEL_INDEX] = 0;
                    this_kernel_grad[ji_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * CONSTANT_LEVEL_INDEX] = 0;
                }
                if (RBF_LEVEL_INDEX > -1) {
                    this_kernel_grad[ij_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * RBF_LEVEL_INDEX] = 0;
                    this_kernel_grad[ji_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * RBF_LEVEL_INDEX] = 0;
                }
                #ifdef IS_VEC_RBF_SCALE
                for (int i_scale=RBF_SCALE_VEC_INDEX_0; i_scale < RBF_SCALE_VEC_INDEX_1; i_scale++) {
                    this_kernel_grad[ij_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * i_scale] = 0;
                    this_kernel_grad[ji_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * i_scale] = 0;
                }
                #else
                if (RBF_SCALE_INDEX > -1) {
                    this_kernel_grad[ij_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * RBF_SCALE_INDEX] = 0;
                    this_kernel_grad[ji_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * RBF_SCALE_INDEX] = 0;
                }
                #endif
            } else {
                DATA_TYPE distance = 0;
                #ifdef IS_VEC_RBF_SCALE
                for (int i_feat=0; i_feat<NUM_FEATURES; i_feat++) {
                    tmp_dist[i_feat] = ((i_data_point[i_feat] - j_data_point[i_feat])*(i_data_point[i_feat] - j_data_point[i_feat]))/(
                        2.0f * rbf_length_scales[i_feat] * rbf_length_scales[i_feat]);
                    distance += tmp_dist[i_feat];
                }
                #else
                for (int i_feat=0; i_feat<NUM_FEATURES; i_feat++) {
                    tmp_dist[i_feat] = ((i_data_point[i_feat] - j_data_point[i_feat])*(i_data_point[i_feat] - j_data_point[i_feat]));
                    distance += tmp_dist[i_feat];
                }
                distance = distance / (2.0f * rbf_length_scales[0] * rbf_length_scales[0]);
                #endif
                exp_value = EXP(-distance) * rbf_level;
                kernel_value = exp_value + const_value;

                //if (distance == 0) {
                if (i_nn == j_nn) {
                    kernel_value += white_value;
                    if (WHITE_NOISE_INDEX > -1) {
                        this_kernel_grad[ij_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * WHITE_NOISE_INDEX] = white_value;
                        this_kernel_grad[ji_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * WHITE_NOISE_INDEX] = white_value;
                    }
                }
                if (CONSTANT_LEVEL_INDEX > -1) {
                    this_kernel_grad[ij_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * CONSTANT_LEVEL_INDEX] = const_value;
                    this_kernel_grad[ji_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * CONSTANT_LEVEL_INDEX] = const_value;
                }
                if (RBF_LEVEL_INDEX > -1) {
                    this_kernel_grad[ij_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * RBF_LEVEL_INDEX] = exp_value;
                    this_kernel_grad[ji_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * RBF_LEVEL_INDEX] = exp_value;
                }
                #ifdef IS_VEC_RBF_SCALE
                for (int i_scale=RBF_SCALE_VEC_INDEX_0; i_scale < RBF_SCALE_VEC_INDEX_1; i_scale++) {
                    int tmp = i_scale - RBF_SCALE_VEC_INDEX_0;
                    this_kernel_grad[ij_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * i_scale] = exp_value * 2 * tmp_dist[tmp];
                    this_kernel_grad[ji_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * i_scale] = exp_value * 2 * tmp_dist[tmp];
                }
                #else
                if (RBF_SCALE_INDEX > -1) {
                    this_kernel_grad[ij_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * RBF_SCALE_INDEX] = exp_value * 2 * distance;
                    this_kernel_grad[ji_index * PITCH_NN_KERNEL_GRAD + THETA_PITCH * RBF_SCALE_INDEX] = exp_value * 2 * distance;
                }
                #endif
            }
            this_kernel[ij_index * PITCH_NN_KERNEL] = kernel_value;
            this_kernel[ji_index * PITCH_NN_KERNEL] = kernel_value;
        }
    }
}
"""
