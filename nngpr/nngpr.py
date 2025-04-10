from operator import itemgetter
from numbers import Integral

import numpy as np
from scipy import linalg as sl
from scipy import sparse as sp
from sklearn.gaussian_process import GaussianProcessRegressor as SKGaussianProcessRegressor
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.parallel import Parallel, delayed
from sklearn.gaussian_process import kernels
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn import base as sk_base


GPR_CHOLESKY_LOWER = True


class NNGPR(SKGaussianProcessRegressor):
    """Implements Nearest Neighbor Gaussian Process Regressor according to Datta 2016 (https://arxiv.org/abs/1406.7343).
    In a nutshell, this model works by building a local Gaussian Process around the nearest neighbors of a given point. 
    NNGPR overcomes quadratic complexity of the standard Gaussian Processes. The complexity of a NNGPR with M nearest 
    neighbors is N*M^2 for the Gaussian Process part (kernel and matrix operations, usually the bottleneck), and N^2 for 
    the nearest neighbors search. 
    Moreover, NNGPR does not have a quadratic memory usage since it never stores the full kernel or covariance matrix,
    thus allows to use the model on large datasets.

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

    batch_size : int, default 500
        Batch size used to split the calculation in batches. Large batch size may cause out of memory errors. Low batch sizes may prevent
        parallelism exploitation.


    Attributes
    ----------
    X_train_ : see sklearn.gaussian_process.GaussianProcessRegressor

    y_train_ : see sklearn.gaussian_process.GaussianProcessRegressor

    kernel_ : see sklearn.gaussian_process.GaussianProcessRegressor

    num_nn : int, number of nearest neighbors

    n_jobs : int | None, number of parallel jobs to run for fit, predict or sampling

    """

    _parameter_constraints: dict = {
        **SKGaussianProcessRegressor._parameter_constraints,
        "num_nn": [Interval(Integral, 1, None, closed="left")],
        "n_jobs": [Integral, None],
        "nn_type": [StrOptions({"kernel-space", "input-space"})],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
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
            batch_size=500):

        super().__init__(
            kernel=kernel, alpha=alpha, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y, copy_X_train=copy_X_train, random_state=random_state)

        # Store inputs
        self.num_nn = num_nn
        self.n_jobs = n_jobs
        self.nn_type = nn_type
        self.batch_size = batch_size

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

        return self

    def find_nearest_neighbours(self, kernel, num_nn, ref, query, ref_cut_index=-1):
        """Finds the nearest neighbors in 'ref' for each point in 'query'.

        Parameters
        ----------
        kernel : kernels.Kernel
            Kernel of the gaussian process

        num_nn : int
            number of nearest  neighbors to use

        ref : array-like of shape (n_reference_points, n_features)
            Search points for the nearest neighbors search

        query : array-like of shape (n_query_points, n_features)
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

        ref_cut_index = ref.shape[0] if ref_cut_index < 0 else ref_cut_index
        nn_indices = -np.ones((query.shape[0], num_nn), dtype=np.int32)

        if self.nn_type == 'kernel-space':
            for i in range(max(0, 1 - ref_cut_index), nn_indices.shape[0]):  # Starting point ensures positive tmp variable below
                tmp = min(i + ref_cut_index, num_nn)
                local_kern = kernel(query[i].reshape((1, -1)), ref[:i + ref_cut_index])
                nn_indices[i, :tmp] = np.argsort(-local_kern[0, :])[:tmp]
        else:
            for i in range(max(0, 1 - ref_cut_index), nn_indices.shape[0]):  # Starting point ensures positive tmp variable below
                tmp = min(i + ref_cut_index, num_nn)
                nn = NearestNeighbors(n_neighbors=tmp)
                nn.fit(ref[:i + ref_cut_index])
                nn_indices[i, :tmp] = nn.kneighbors(query[i].reshape((1, -1)), return_distance=False)

        return nn_indices

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

        # Define x train
        x_full = self.X_train_
        nt = x_full.shape[0]
        n_theta = len(theta)

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
                kernel, min(self.num_nn, i1 - 1), x_full[:i1], x_full[i0:i1], ref_cut_index=i0)
            nn_indices = np.concatenate([
                np.arange(i0, i0 + nn_indices.shape[0]).reshape((-1, 1)), nn_indices], axis=1)

            # Evaluates the kernel and kernel gradient
            K, K_gradient = self._fill_nn_kernel(
                x_full, nn_indices, kernel, eval_gradient=eval_gradient)

            # Add jitter to the kernel
            ind = np.where(nn_indices > -1)
            K[ind[0], ind[1], ind[1]] += self.alpha
            if eval_gradient:
                K_gradient = np.moveaxis(K_gradient, -1, 0)  # Move the axis corresponding to theta at the beginning

            n = K.shape[0]

            # Calculate the Cholesky decomposition
            L = self.batched_chofactor(K[:, 1:, 1:], GPR_CHOLESKY_LOWER, overwrite_x=False)

            # Calculate the y train
            y_train = y_train_full[i0:i1]
            y_train_nn = self._build_ytrain_given_nn_indices(y_train_full, nn_indices)
            del nn_indices

            # Now calculate the log marginal likelihood

            # Define matrices K_xn and K_nn_inv
            K_xn = K[:, 0, 1:].reshape((n, 1, K.shape[1] - 1))
            def K_nn_inv(right, add_dim=False):
                if add_dim:
                    return self.batched_chosolve(L[np.newaxis, :], right, GPR_CHOLESKY_LOWER)
                return self.batched_chosolve(L, right, GPR_CHOLESKY_LOWER)
                
            mu = (K_xn @ K_nn_inv(y_train_nn[:, 1:])).reshape(y_train.shape)
            sigma = np.sqrt(K[:, 0, 0].reshape((n, 1)) - (K_xn @ K_nn_inv(np.swapaxes(K_xn, 1, 2))
                                                          ).reshape((n, 1)))
            this_log_lkl = -0.5 * (y_train - mu) ** 2 / sigma ** 2 - 0.5 * np.log(2 * np.pi) - np.log(sigma)

            # the log likehood is sum-up across the outputs and the first dimension
            log_likelihood = this_log_lkl.sum(axis=(0, -1))

            if eval_gradient:
                # Expand quantities by adding the dimension corresponding to theta
                sigma, mu, y_train = sigma[np.newaxis, :], mu[np.newaxis, :], y_train[np.newaxis, :]

                # Derivative of K_nn
                dK_nn_inv_dtheta = lambda right: -K_nn_inv(
                    K_gradient[:, :, 1:, 1:] @ K_nn_inv(right)[np.newaxis, :], add_dim=True)
                # Derivative of K_xn
                dK_xn_dtheta = K_gradient[:, :, 0, 1:].reshape((n_theta, n, 1, K.shape[1] - 1))
                # Derivative of mu
                dmu_dtheta = (dK_xn_dtheta @ K_nn_inv(y_train_nn[:, 1:])[np.newaxis, :]).reshape(
                    (n_theta, *y_train.shape[1:])) + \
                    (K_xn[np.newaxis, :] @ dK_nn_inv_dtheta(y_train_nn[:, 1:])).reshape((n_theta, *y_train.shape[1:]))

                # Derivarive of sigma
                dsigma_dtheta = 0.5 / sigma * (
                        K_gradient[:, :, 0, 0].reshape((n_theta, n, 1)) -
                        2 * (dK_xn_dtheta @ (K_nn_inv(np.swapaxes(K_xn, 1, 2)))[np.newaxis, :]).reshape(
                            (n_theta, n, 1)) - (K_xn[np.newaxis, :] @ dK_nn_inv_dtheta(np.swapaxes(
                                K_xn, 1, 2))).reshape((n_theta, n, 1)))

                log_likelihood_gradient = (-1 / sigma + (y_train - mu) ** 2 / sigma ** 3) * dsigma_dtheta + (
                            y_train - mu) / sigma ** 2 * dmu_dtheta

                log_likelihood_gradient = np.sum(
                    log_likelihood_gradient, axis=(1, 2))  # Axis 0 is the theta parameter, axis 2 is the dimension of the output

            else:
                log_likelihood_gradient = np.zeros(n_theta)

            return log_likelihood, log_likelihood_gradient

        num_batches = int(np.ceil(nt / self.batch_size))
        batch_results = Parallel(self.n_jobs)(delayed(run_batch)(i) for i in range(num_batches))
        log_likelihood = sum(x[0] for x in batch_results)

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

        random_state : int, RandomState instance or None, default=0
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

        rng = check_random_state(random_state)

        # If conditioning_method is only-train, then each sample is independent of the others and we can use
        # a faster (full parallel) algorithm
        if conditioning_method == 'only-train':
            mu, sigma = self.predict(X, return_std=True)
            mu, sigma = mu[..., np.newaxis], sigma[..., np.newaxis]
            if y_dim == 1:
                mu, sigma = mu[..., np.newaxis], sigma[..., np.newaxis]
            y_samples = rng.normal(loc=mu, scale=sigma, size=(nq, y_dim, n_samples))

            if y_dim == 1:
                y_samples = y_samples[:, 0, :]  # Remove dimension corresponding to the y-dimension

            return y_samples
        
        # If conditioning_method is not only-train, continue here with the sequential algorithm

        # Find nearest neighbours. They could be searched in batches, but not really needed since the memory footprint is 
        # negligible. The biggest memory usage comes from the kernel K and its Cholesky decomposition.
        nn_indices, x_full = self._find_nn_indices_for_train_and_eval(
            kernel, x_train, X, condition_on_eval=conditioning_method != 'only-train',
            double_nn=conditioning_method == 'full-double-nn')

        # Allocate output and temporary vars
        y_samples = np.ones((nq, y_dim, n_samples)) * np.nan
        y_nn = np.empty((nn_indices.shape[1] - 1, y_dim, n_samples))

        # Loop over batches of data in case the entire arrays cannot be all stored in memory
        for i_batch in range(int(np.ceil(nq / self.batch_size))):
            i0 = self.batch_size * i_batch
            i1 = min(self.batch_size * (i_batch + 1), nq)

            # Evaluates the kernel and kernel gradient
            K, _ = self._fill_nn_kernel(
                x_full, nn_indices[i0:i1], kernel, eval_gradient=False)

            # Add jitter to the kernel
            ind = np.where((nn_indices[i0:i1] < nt) & (nn_indices[i0:i1] >= 0))
            K[ind[0], ind[1], ind[1]] += self.alpha
            del ind

            # Calculate the Cholesky decomposition
            L = self.batched_chofactor(K[:, 1:, 1:], GPR_CHOLESKY_LOWER, overwrite_x=False)

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
                this_K_nn_inv = lambda right: sl.cho_solve((L[i], GPR_CHOLESKY_LOWER), right, overwrite_b=False)

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
        
        # Find nearest neighbours. They could be searched in batches, however when the conditioning_method is different than 'only_train',
        # there is little gain in memory usage. Moreover, biggest memory usage comes from the kernel K and its Cholesky decomposition.
        # TODO: this could be made parallel
        nn_indices, x_full = self._find_nn_indices_for_train_and_eval(
            kernel, x_train, X, condition_on_eval=conditioning_method != 'only-train',
            double_nn=conditioning_method == 'full-double-nn')

        # If conditioning_method is only-train then each sample is independent of the others. Thus, if return_cov is False, we can use
        # a faster (full parallel) algorithm
        if (conditioning_method == 'only-train') and (not return_cov):

            # Loop over batches of data in case the entire arrays cannot be all stored in memory
            def run_batch(i_batch):
                i0 = self.batch_size * i_batch
                i1 = min(self.batch_size * (i_batch + 1), nq)

                # Evaluates the kernel and kernel gradient
                K, _ = self._fill_nn_kernel(
                    x_full, nn_indices[i0:i1], kernel, eval_gradient=False)
                # Add jitter to the kernel
                ind = np.where((nn_indices[i0:i1] < nt) & (nn_indices[i0:i1] >= 0))
                K[ind[0], ind[1], ind[1]] += self.alpha
                del ind

                # Calculate y_train_nn
                y_train_nn = self._build_ytrain_given_nn_indices(y_train, nn_indices[i0:i1, 1:])

                # Calculate the Cholesky decomposition
                L = self.batched_chofactor(K[:, 1:, 1:], GPR_CHOLESKY_LOWER, overwrite_x=False)

                # Define relevant matrices
                K_xn = K[:, :1, 1:]
                def K_nn_inv(right):
                    return self.batched_chosolve(L, right, GPR_CHOLESKY_LOWER)
                
                # Calculate mean
                mean = (K_xn @ K_nn_inv(y_train_nn))[:, 0, :]

                if return_std:
                    n = i1 - i0
                    std = np.sqrt(K[:, 0, 0].reshape((n, 1)) - (K_xn @ K_nn_inv(np.swapaxes(K_xn, 1, 2))
                                                          ).reshape((n, 1))).reshape(-1)
                else:
                    std = None

                return mean, std

            num_batches = int(np.ceil(nq / self.batch_size))
            batch_results = Parallel(self.n_jobs)(delayed(run_batch)(i) for i in range(num_batches))
            mean = np.concatenate([x[0] for x in batch_results], axis=0)

            # Return output
            mean = format_mean(mean)
            if return_std:
                std = np.concatenate([x[1] for x in batch_results], axis=0)
                return mean, format_sigma(std)
            return mean

        # If we reach this point, we have to go through the slow, sequential algorithm
        nn_indices[:, 1:] = np.sort(nn_indices[:, 1:], axis=1)  # To access partial covariance elements less randomly
        num_nn = nn_indices.shape[1]

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
            K, _ = self._fill_nn_kernel(
                x_full, nn_indices[i0:i1], kernel, eval_gradient=False)

            # Add jitter to the kernel
            ind = np.where((nn_indices[i0:i1] < nt) & (nn_indices[i0:i1] >= 0))
            K[ind[0], ind[1], ind[1]] += self.alpha
            del ind

            # Calculate the Cholesky decomposition
            L = self.batched_chofactor(K[:, 1:, 1:], GPR_CHOLESKY_LOWER, overwrite_x=False)

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
                this_K_nn_inv = lambda right: sl.cho_solve((L[i], GPR_CHOLESKY_LOWER), right, overwrite_b=False)

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
    def batched_chosolve(L, y, lower):
        """Solves a batch of linear systems A_i * X_i = B_i given the Cholesky decomposition L_i of the symmetric matrices A_i.

        Parameters
        ----------

        L : np.ndarray of shape (..., n, n)
            Batch of Cholesky decomposition of the symmetric matrices.

        y : np.ndarray of shape (..., n, nrhs)
            Vector arrays B_i that from the rhs of A_i * X_i = B_i. In case nrhs=1, the last axis can be squeezed.

        lower : bool
            If True, then the Cholesky decomposition is stored in the lower triangular part of L_i, else in the upper triangular part.

        Returns
        -------

        x : np.ndarray of the same shape of input y
            Solution arrays X_i of A_i * X_i = B_i.

        """
        assert len(L.shape) > 2
        assert L.shape[-2] == L.shape[-1], "Not square matrices"
        assert len(y.shape) == len(L.shape)
        if ((y.shape[0] == 1) and (L.shape[0] > 1)) or ((y.shape[0] > 1) and (L.shape[0] == 1)):  # Means theta has multiple dimensions
            assert (L.shape[1:-1] == y.shape[1:-1]), "L and y shapes mismatch"
        else:
            assert L.shape[:-1] == y.shape[:-1], "L and y shapes mismatch"

        shape = L.shape[:-2] if y.shape[0] == 1 else (y.shape[0], *L.shape[1:-2])
        out = np.empty_like(y)

        L_loop_shape = L.shape[:-2]
        L_loop_size = np.prod(L_loop_shape)
        y_loop_shape = y.shape[:-2]
        y_loop_size = np.prod(y_loop_shape)
        for i in range(np.prod(shape)):
            index = np.unravel_index(i, shape)
            y_index = np.unravel_index(i % y_loop_size, y_loop_shape)
            L_index = np.unravel_index(i % L_loop_size, L_loop_shape)
            out[index] = sl.cho_solve((L[L_index], lower), y[y_index], overwrite_b=False)

        return out
    
    @staticmethod
    def batched_chofactor(x, lower, overwrite_x=False) -> np.ndarray:
        """Calculates the Cholesky decomposition for a batch of symmetric and square matrices x_i.

        Parameters
        ----------

        x : np.ndarray of shape (..., n, n)
            Batch of symmetric matrices for which the Cholesky decomposition is calculated.

        lower : bool
            If True, then the Cholesky decomposition is stored in the lower triangular part of L_i, else in the upper triangular part.
            Note that the triangular part which does not store the cholesky decomposition does not contain valid numbers (i.e. it may be
            not zeroed).

        overwrite_x : bool, default False
            If True, the Cholesky decomposition may be stored in the same input array x to avoid new memory allocation.

        Returns
        -------

        L : np.ndarray of the same shape of input x
            Batch of Cholesky decomposition of input arrays x.

        """
        
        assert len(x.shape) > 2, "Not enough dimensions"
        assert x.shape[-2] == x.shape[-1], "Not square matrices"

        shape = x.shape[:-2]
        out = x if overwrite_x else np.empty_like(x)
        for i in range(np.prod(shape)):
            index = np.unravel_index(i, shape)
            out[index] = sl.cho_factor(x[index], lower=lower, overwrite_a=overwrite_x)[0]
        return out
    
    @staticmethod
    def _build_ytrain_given_nn_indices(y_train, nn_indices):
        """Calculates the array containing the observed y values for each nearest neighbor.

        Parameters
        ----------

        y_train : np.ndarray of shape (n_train,)
            Observed y values in the training dataset.

        nn_indices : np.ndarray of shape (n_query, n_nn)
            Array that at position [i, j] stores the index of the j-th nearest neighbor for the i-th query point. nn_indices[i, j]
            must be set to -1 when the j-th nearest neighbor for the i-th query point is not defined.

        Returns
        -------

        y_train_nn : np.ndarray of the same shape (n_query, n_nn)
            Array that at position [i, j] stores the observed y_train on the j-th nearest neighbor for the i-th query point. If the j-th
            nearest neighbor index in nn_indices is -1, then y_train_nn[i, j] is set to zero.

        """

        y_train_nn = np.empty((*nn_indices.shape, y_train.shape[-1]), y_train.dtype)
        usable = nn_indices != -1
        y_train_nn[usable, :] = y_train[nn_indices[usable], :]
        y_train_nn[~usable, :] = 0

        return y_train_nn

    @staticmethod
    def _fill_nn_kernel(x, nn_indices, kernel, eval_gradient=False):
        """Calculates the kernel based on nearest neighbors given the nearest neighbors indices.

        Parameters
        ----------

        x : np.ndarray of shape (n_points, n_features)
            Input dataset

        nn_indices : np.ndarray of shape (n_query, n_nn)
            Array that at position [i, j] stores the index of the j-th nearest neighbor for the i-th query point. nn_indices[i, j]
            must be set to -1 when the j-th nearest neighbor for the i-th query point is not defined.

        kernel : kernels.Kernel
            Kernel of the gaussian process

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
        nn = nn_indices.shape[-1]

        # Allocate output
        nn_kernel = np.empty((nn_indices.shape[0], nn, nn), dtype=x.dtype)
        nn_kernel_grad = np.empty((nn_indices.shape[0], nn, nn, len(kernel.theta)), dtype=x.dtype) if eval_gradient else np.empty(0)

        # Loop over every point
        for i in range(nn_kernel.shape[0]):
            ind = nn_indices[i]
            is_negative = ind == -1
            res = kernel(x[nn_indices[i]], eval_gradient=eval_gradient)
            if eval_gradient:
                nn_kernel[i] = res[0]
                nn_kernel_grad[i] = res[1]
                # Remove entries corresponding to negative indices
                nn_kernel_grad[i, is_negative, :] = 0
                nn_kernel_grad[i, :, is_negative] = 0
            else:
                nn_kernel[i] = res

            # Remove entries corresponding to negative indices
            nn_kernel[i, is_negative, :] = 0
            nn_kernel[i, :, is_negative] = 0
            nn_kernel[i, is_negative, is_negative] = 1
        
        return nn_kernel, nn_kernel_grad

    def _find_nn_indices_for_train_and_eval(self, kernel, x_train, x_query, condition_on_eval, double_nn):
        """Calculates the array containing the observed y values for each nearest neighbor.

        Parameters
        ----------

        kernel : kernels.Kernel
            Kernel of the gaussian process

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

        nn_indices : np.ndarray of shape (n_query, n_nn + 1) or (n_query, n_nn*2 + 1)
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
                    nn_ind_train = self.find_nearest_neighbours(kernel, min(self.num_nn, x_train.shape[0]), x_train, x_query)
                else:
                    nn_ind_train = np.empty((0, 0)), np.empty((0, 0), dtype=np.int32)
                nn_ind_nontrain = self.find_nearest_neighbours(
                    kernel, min(self.num_nn, x_query.shape[0] - 1), x_query, x_query, ref_cut_index=0)
                nn_ind_nontrain[nn_ind_nontrain != -1] += x_train.shape[0]
                if x_train.shape[0] > 0:
                    # Need to combine nn_ind_train and nn_ind_nontrain
                    nn_indices = np.empty((nn_ind_train.shape[0], nn_ind_train.shape[1] + nn_ind_nontrain.shape[1]),
                                          dtype=nn_ind_train.dtype)
                    arange = np.tile(np.arange(nn_indices.shape[1], dtype=np.int32), (nn_indices.shape[0], 1))

                    # First insert indices from indices_0
                    is_valid = nn_ind_train != -1
                    n_to_add_0 = np.sum(is_valid, axis=1).reshape((-1, 1))
                    nn_indices[arange < n_to_add_0] = nn_ind_train[is_valid]

                    # Then insert indices from indices_1
                    is_valid = nn_ind_nontrain != -1
                    n_to_add_1 = n_to_add_0 + np.sum(is_valid, axis=1).reshape((-1, 1))
                    nn_indices[(arange >= n_to_add_0) & (arange < n_to_add_1)] = nn_ind_nontrain[is_valid]

                    # Set the rest to the null value
                    nn_indices[arange >= n_to_add_1] = -1
                else:
                    nn_indices = nn_ind_nontrain
                del nn_ind_train, nn_ind_nontrain
            else:
                x_full = get_x_full()
                nn_indices = self.find_nearest_neighbours(
                    kernel, min(self.num_nn, x_full.shape[0]), x_full, x_query, ref_cut_index=x_train.shape[0])
        else:
            num_nn = min(x_train.shape[0], self.num_nn)
            if num_nn > 0:
                nn_indices = self.find_nearest_neighbours(kernel, num_nn, x_train, x_query)
            else:
                nn_indices = np.empty((x_query.shape[0], 0), dtype=np.int32)

        nn_indices = np.concatenate([
            np.arange(x_train.shape[0], x_train.shape[0] + nn_indices.shape[0]).reshape((-1, 1)), nn_indices], axis=1)
        x_full = get_x_full() if x_full is None else x_full
        return nn_indices, x_full
