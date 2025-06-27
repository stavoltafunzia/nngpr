""" This module implements core functionalities for the Nearest Neighbors Gaussian Process Regressor. """

import os
from operator import itemgetter
from numbers import Integral
from abc import abstractmethod, ABC
import multiprocessing
from queue import Empty
import collections
import itertools

import numpy as np
from scipy import linalg as sl
from scipy import sparse as sp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn import base as sk_base
from sklearn.utils.parallel import Parallel, delayed

from . import batched_kernels


class Nngpr(GaussianProcessRegressor, ABC):
    """Implements Nearest Neighbor Gaussian Process Regressor according to Datta 2016 (https://arxiv.org/abs/1406.7343).
    In a nutshell, this model works by building a local Gaussian Process around the nearest neighbors of a given point. 
    Nngpr overcomes quadratic complexity of the standard Gaussian Processes. The complexity of a Nngpr with M nearest 
    neighbors is N*M^2 for the Gaussian Process part (kernel and matrix operations, usually the bottleneck), and N^2 for 
    the nearest neighbors search. 
    Moreover, Nngpr does not have a quadratic memory usage since it never stores the full kernel or covariance matrix,
    thus allows to use the model on large datasets.

    It is built on top of sklearn GaussianProcessRegressor, maintaining similar apis.

    End users are not supposed to use this class directly, rather other helpers provided in nngpr.numpy_nngpr, 
    nngpr.cupy_nngpr, nngpr.torch_nngpr. Use this class if you know what you are doing.


    Parameters
    ----------

    workers : List[NngprWorker]
        List of workers to be used for the actual calculations. Workers can eventually be of different type and performance.

    kernel : None or nngpr.batched_kernels.BatchedKernel
        Kernel of the gaussian process

    alpha : see sklearn.gaussian_process.GaussianProcessRegressor

    optimizer : see sklearn.gaussian_process.GaussianProcessRegressor

    n_restarts_optimizer : see sklearn.gaussian_process.GaussianProcessRegressor

    normalize_y : see sklearn.gaussian_process.GaussianProcessRegressor

    copy_X_train : see sklearn.gaussian_process.GaussianProcessRegressor

    n_targets : see sklearn.gaussian_process.GaussianProcessRegressor

    random_state : see sklearn.gaussian_process.GaussianProcessRegressor

    num_nn : int | None, default 32
        Number of nearest neighbors to use. If None, the default value is used.

    nn_type : str, default 'kernel-space'
        Search space for the nearest neighbors. Can be either 'kernel-space' or 'input-space'. If 'kernel-space' nearest neighbors
        are searched in the kernel space, i.e. the neighbors of a query point are the points with the highest covariance w.r.t. the 
        query point. When 'input-space' nearest neighbors are searched in the input feature space, using euclidean distance.

    batch_size : int | None
        Batch size used to split the calculation in batches. Large batch size may cause out of memory errors. Low batch sizes may prevent
        parallelism exploitation. That this batch size is only used to split the total work into chunks that are submitted to the
        workers. Note that each worker may have assigned a different batch size: in case the worker batch size is smaller than this batch
        size, the actual job performed by the worker is further split according to the worker-defined batch size. If None, some euristic
        rule is used to calculate the batch size depending on worker characteristics.

    allow_downcast_f32 : str | None, default 'only-nn'
        Specifies if input arrays can be downcasted to float32. This can be useful in case workers are implemented in consumer-grade GPUs
        which come with crippled float64 performance. Must be either 'no', 'yes' or 'only-nn'. If 'no', no downcast is performed. When 
        'only-nn', downcast is done only for the nearest neighbors search. If 'yes', then downcast is done for all operations (including 
        matrix inverse operations). When 'yes', this may negatively affect convergence of the optimisation problem when fitting. It is 
        recommended to set to 'only-nn' when using GPUs with poor float64 performance. If None the default value is used.

    distribute_method : str | None, defaults to 'multiprocessing' if 'fork' is available else 'joblib'
        Specifies the backend to use to distribute the work across the workers when more than one worker is used. Must one of the 
        following: 'joblib', 'multiprocessing', 'multiprocessing-heterogenous', 'sequential'. When None, 'joblib' is used if the 'fork'
        start method is not available in multiprocessing, otherwise 'multiprocessing' is used. 'joblib' uses the
        joblib (from sklearn package, thus with sklearn configuration) to dispatch tasks among the workers. When 'multiprocessing', Python 
        multiprocessing is used. Both in the case of 'joblib' and 'multiprocessing', tasks are split evenly among workers. In case workers 
        are heterogeneous, with different performance, this is suboptimal with the slowest worker determining the runtime and faster 
        workers idling for part of the runtime. 'multiprocessing-heterogenous' addresses this, it uses Python multiprocessing together 
        with a Queue to dispatch tasks to the workers as soon as they are ready to pick up one. Using 'multiprocessing-heterogenous' is 
        likely to cause random samples to be random even when fixed seeds are used. This happens because different workers are likely
        to pick up different tasks on each run. When 'sequential' is used, tasks are executed sequentially without
        parallel computiung. Note this argument is not relevant when only one worker is used.

    mp_start_method : str | None, default 'fork' if available, else 'spawn'
        Multiprocessing start method to use when 'joblib' is not used as distribute_method. Must be one of the methods returned by
        multiprocessing.get_all_start_methods().

    random_gen_batch_byte_size : int | None, default to 1024**3
        Batch size (in bytes) used when drawing samples from the random normal distrubution. Sampling is divided in batches of 
        random_gen_batch_byte_size size and then distributed accross the workers. If None, defaults to 1024**3 (1GB)

    distribute_environ : dict[str, str] | None
        Environment variables to be set before spawning subprocess for parallel calculation. Allows to avoid overparallelization by
        Blas/Mkl libraries. If None, defaults to the following enviornment variables:
        {
            'OPENBLAS_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OMP_NUM_THREADS': '1',
        }
        Variables are restored after parallel calculation is completed.

    Attributes
    ----------
    X_train_ : see sklearn.gaussian_process.GaussianProcessRegressor

    y_train_ : see sklearn.gaussian_process.GaussianProcessRegressor

    kernel_ : see sklearn.gaussian_process.GaussianProcessRegressor

    """

    _parameter_constraints: dict = {
        **GaussianProcessRegressor._parameter_constraints,
        "num_nn": [Interval(Integral, 1, None, closed="left")],
        "nn_type": [StrOptions({"kernel-space", "input-space"})],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "allow_downcast_f32": [StrOptions({"yes", "no", "only-nn"})],
        "distribute_method": [StrOptions({"joblib", "multiprocessing", "multiprocessing-heterogenous", "sequential"})],
        "mp_start_method": [StrOptions(set(multiprocessing.get_all_start_methods()))],
        "random_gen_batch_byte_size": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
            self,
            workers,
            kernel=None,
            *,
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=0,
            normalize_y=False,
            copy_X_train=True,
            random_state=None,
            num_nn=None,
            nn_type=None,
            batch_size=None,
            allow_downcast_f32=None,
            distribute_method=None,
            mp_start_method=None,
            random_gen_batch_byte_size=None,
            distribute_environ=None):

        super().__init__(
            kernel=kernel, alpha=alpha, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y, copy_X_train=copy_X_train, random_state=random_state)

        # Store inputs
        self.workers = workers
        self.num_nn = 32 if num_nn is None else num_nn
        self.nn_type = 'kernel-space' if nn_type is None else nn_type
        self.allow_downcast_f32 = "only-nn" if allow_downcast_f32 is None else allow_downcast_f32
        is_fork_available = 'fork' in multiprocessing.get_all_start_methods()
        self.distribute_method = ('multiprocessing' if is_fork_available else 'joblib') if distribute_method is None else distribute_method
        self.mp_start_method = ('fork' if is_fork_available else 'spawn') \
            if mp_start_method is None else mp_start_method
        self.random_gen_batch_byte_size = 1024**3 if random_gen_batch_byte_size is None else random_gen_batch_byte_size
        self.distribute_environ = distribute_environ
        if self.distribute_environ is None:
            self.distribute_environ = {
                'OPENBLAS_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1',
                'OMP_NUM_THREADS': '1',
            }
        self.batch_size = batch_size
        if self.batch_size is None:
            workers_bs = [w.nn_large_size_preferred for w in self.workers]
            workers_bs = [x for x in workers_bs if x is not None]
            if bool(workers_bs):
                self.batch_size = max(1, min(workers_bs))
            else:
                workers_bs = [w.batch_size_for_nn(self.num_nn, 4 if self.allow_downcast_f32 == 'yes' else 8, kernel) for w in self.workers]
                workers_bs = [x for x in workers_bs if x is not None]
                if bool(workers_bs):
                    self.batch_size = max(1, max(workers_bs))
                else:
                    self.batch_size = 1000  # Some default value

    def distribute(self, function, n, *, worker_arguments=None, common_arguments=None):
        """ Distributes one function across the workers for parallel computation
        
        Parameters
        ----------
        function : callable
            Function to be executed. The function will be executed n times, with the following signature:
                function(i_worker, queue, *worker_arguments[i_worker], *common_arguments)
            where i_worker is the worker index, queue is a Queue-like object whose methods `get` or `get_nowait` should
            be used to retrieve the task index to perform. Only `get` and `get_nowait` are guaranteed to be implemented
            in this Queue-like object. The task index that is retrieved from the queue is in the range 0..n-1. When 
            there are no remaining tasks to be performed, a `get` or `get_nowait` call on the queue will raise an Empty
            exception and the function should return.

        n : int
            Number of times the function will be called

        worker_arguments : List[Iterable]
            Worker-specific arguments passed to the function. Lenght of this list must match the number of workers.

        common_arguments : Iterable
            Arguments passed to the function

        Returns
        -------
        list(Any) : list of length n containing returned values from each function call. 

        """

        assert n > 0, "No tasks to perform, are you sure?"
        nprocs = min(n, len(self.workers))
        worker_arguments = [[]] * nprocs if worker_arguments is None else worker_arguments
        common_arguments = [] if common_arguments is None else common_arguments
        assert len(worker_arguments) >= nprocs, "Number of worker arguments is less than number of workers"

        if (nprocs == 1) or (self.distribute_method == 'sequential'):
            return [function(0, DummyQueue(range(n)), *worker_arguments[0], *common_arguments)]

        # backup env vars
        env_bak = {k: os.environ.get(k) for k in self.distribute_environ}
        for k, v in self.distribute_environ.items():
            os.environ[k] = v

        # Run job in parallel
        if self.distribute_method == 'joblib':
            queues = [DummyQueue(range(i, n, nprocs)) for i in range(nprocs)]
            res = Parallel(nprocs)(delayed(function)(i, queue, *worker_arguments[i], *common_arguments) for i, queue in enumerate(queues))
        
        elif self.distribute_method == 'multiprocessing':
            queues = [DummyQueue(range(i, n, nprocs)) for i in range(nprocs)]
            ctx = multiprocessing.get_context(self.mp_start_method)
            with ctx.Pool(nprocs) as p:
                res = p.starmap(function, ((i, queue, *worker_arguments[i], *common_arguments) for i, queue in enumerate(queues)))
            
        elif self.distribute_method == 'multiprocessing-heterogenous':
            if self.mp_start_method is None:
                ctx = multiprocessing
            else:
                ctx = multiprocessing.get_context(self.mp_start_method)
            m = ctx.Manager()
            q = m.JoinableQueue()
            for x in range(n):
                q.put(x)
            with ctx.Pool(processes=nprocs) as pool:
                res = pool.starmap(function, [(i, q, *worker_arguments[i], *common_arguments) for i in range(nprocs)])

        else:
            raise NotImplementedError(self.distribute_method)

        # Restore env vars
        for k, v in env_bak.items():
            if v is None:
                os.environ.pop(k)
            else:
                os.environ[k] = v

        return res

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
            Nngpr class instance.
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

        for worker in self.workers:
            worker.clear_cache()

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

        # Define x train
        x_train = self.X_train_
        n_theta = len(theta)

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, None]

        # Downcast to float32 if needed
        if self.allow_downcast_f32 == "yes":
            x_train, y_train = x_train.astype('float32', copy=False), y_train.astype('float32', copy=False)

        # Calculate log marginal likelihood in parallel
        num_batches = int(np.ceil(x_train.shape[0] / self.batch_size))
        batch_results = self.distribute(
            self._log_likelihood_worker, 
            num_batches,
            common_arguments=(n_theta, kernel, x_train, y_train, eval_gradient))
        log_likelihood = sum(x[0] for x in batch_results)

        if eval_gradient:
            log_likelihood_gradient = np.sum([x[1] for x in batch_results], axis=0)
            return log_likelihood, log_likelihood_gradient

        return log_likelihood

    def _log_likelihood_worker(self, i_worker, queue, n_theta, kernel, x_train, y_train, eval_gradient):
        """ Calculates the log-likelihood on part of the training data using a specific worker. Every time
        the worker has finished processing a chunk/batch of data, a new batch is retrieved from the queue
        until all tasks have been processed.

        Parameters
        ----------
        i_worker : int
            Index of the worker to use

        queue : Queue like instance
            Queue used to retrieve the tasks to be performed

        n_theta : int
            Number of kernel parameters

        kernel : nngpr.batched_kernels.BatchedKernel
            Kernel to use

        x_train : np.ndarray
            Train dataset to use

        y_train : np.ndarray
            Realizations of the train dataset to use

        eval_gradient : bool
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for the processed part of the training data.

        log_likelihood_gradient : ndarray of shape (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta, limitedly to the processed part of the training dataset.
            Contains reliable values only when eval_gradient is True.
        """
        worker = self.workers[i_worker]
        xp = worker.get_array_module()
        dtype = x_train.dtype
        worker_bs = worker.batch_size_for_nn(self.num_nn, x_train.itemsize, kernel)
        large_nn_batch_size = worker.nn_large_size_preferred
        x_train, y_train_in = xp.from_numpy(x_train), xp.from_numpy(y_train)
        nt = x_train.shape[0]

        log_likelihoods = []
        gradients = [np.zeros(n_theta, dtype=dtype)]

        while True:
            total_size = 0
            chunks = []
            prev_i1 = None
            try:
                # First, get the chunks to be processed.
                while True:
                    i0 = queue.get_nowait() * self.batch_size
                    i1 = min(i0 + self.batch_size, nt)
                    total_size += i1 - i0
                    if prev_i1 == i0:
                        chunks[-1] = (chunks[-1][0], i1)
                    else:
                        chunks.append((i0, i1))
                    prev_i1 = i1
                    if (large_nn_batch_size is None) or (total_size >= large_nn_batch_size) or len(chunks) > 1:
                        break
            except Empty:
                if len(chunks) == 0:
                    # If there are no more chunks to be processed, quit
                    worker.clear_cache()
                    return np.sum(log_likelihoods), np.sum(gradients, axis=0)
            
            # Then, process the chunks
            for i0_main, i1_main in chunks:

                if large_nn_batch_size is not None:
                    # Find nearest neighbours in the large chunk if that's preferred by the worker
                    nn_indices_large = self._find_nearest_neighbors_wrapper(
                        worker, x_train[:i1_main], x_train[i0_main:i1_main], kernel, min(self.num_nn, i1_main - 1), ref_cut_index=i0_main)
                    nn_indices_large = xp.concatenate([
                        xp.arange(i0_main, i0_main + nn_indices_large.shape[0]).reshape((-1, 1)), nn_indices_large], axis=1)

                # Then continue the work in smaller chunks to prevent out of memory in the worker
                for i0 in range(i0_main, i1_main, worker_bs):
                    i1 = min(i0 + worker_bs, i1_main)

                    # Find nearest neighbours if not already done
                    if large_nn_batch_size is None:
                        nn_indices = self._find_nearest_neighbors_wrapper(
                            worker, x_train[:i1], x_train[i0:i1], kernel, min(self.num_nn, i1 - 1), ref_cut_index=i0)
                        nn_indices = xp.concatenate([
                            xp.arange(i0, i0 + nn_indices.shape[0]).reshape((-1, 1)), nn_indices], axis=1)
                    else:
                        nn_indices = nn_indices_large[i0-i0_main:i1-i0_main]
                        
                    # Evaluates the kernel and kernel gradient
                    K, K_gradient = worker.fill_nn_kernel(
                        x_train, nn_indices, kernel, eval_gradient=eval_gradient)

                    # Add jitter to the kernel
                    ind = xp.where(nn_indices > -1)
                    K[ind[0], ind[1], ind[1]] += self.alpha
                    del ind
                    if eval_gradient:
                        K_gradient = xp.moveaxis(K_gradient, -1, 0)  # Move the axis corresponding to theta at the beginning

                    n = K.shape[0]

                    # Calculate the Cholesky decomposition
                    L, L_lower = worker.batched_chofactor(K[:, 1:, 1:], may_overwrite_x=False)

                    # Calculate the y train
                    y_train = y_train_in[i0:i1]
                    y_train_nn = self._build_ytrain_given_nn_indices(worker, y_train_in, nn_indices)
                    del nn_indices

                    # Define matrices K_xn and K_nn_inv
                    K_xn = K[:, 0, 1:].reshape((n, 1, K.shape[1] - 1))
                    def K_nn_inv(right, add_dim=False):
                        if add_dim:
                            return worker.batched_chosolve(L[None, :], right, L_lower)
                        return worker.batched_chosolve(L, right, L_lower)
                        
                    mu = (K_xn @ K_nn_inv(y_train_nn[:, 1:])).reshape(y_train.shape)
                    sigma = xp.sqrt(K[:, 0, 0].reshape((n, 1)) - (K_xn @ K_nn_inv(xp.swapaxes(K_xn, 1, 2))
                                                                    ).reshape((n, 1)))
                    this_log_lkl = -0.5 * (y_train - mu) ** 2 / sigma ** 2 - 0.5 * np.log(2 * np.pi) - xp.log(sigma)

                    # the log likehood is sum-up across the outputs and the first dimension
                    log_likelihoods.append(xp.to_numpy(this_log_lkl.sum(axis=(0, -1))))

                    if eval_gradient:
                        # Expand quantities by adding the dimension corresponding to theta
                        sigma, mu, y_train = sigma[None, :], mu[None, :], y_train[None, :]

                        # Derivative of K_nn
                        dK_nn_inv_dtheta = lambda right: -K_nn_inv(
                            K_gradient[:, :, 1:, 1:] @ K_nn_inv(right)[None, :], add_dim=True)
                        # Derivative of K_xn
                        dK_xn_dtheta = K_gradient[:, :, 0, 1:].reshape((n_theta, n, 1, K.shape[1] - 1))
                        # Derivative of mu
                        dmu_dtheta = (dK_xn_dtheta @ K_nn_inv(y_train_nn[:, 1:])[None, :]).reshape(
                            (n_theta, *y_train.shape[1:])) + \
                            (K_xn[None, :] @ dK_nn_inv_dtheta(y_train_nn[:, 1:])).reshape((n_theta, *y_train.shape[1:]))
                        del y_train_nn  # Free some memory, especially for GPU

                        # Derivarive of sigma
                        dsigma_dtheta = 0.5 / sigma * (
                                K_gradient[:, :, 0, 0].reshape((n_theta, n, 1)) -
                                2 * (dK_xn_dtheta @ (K_nn_inv(xp.swapaxes(K_xn, 1, 2)))[None, :]).reshape(
                                    (n_theta, n, 1)) - (K_xn[None, :] @ dK_nn_inv_dtheta(xp.swapaxes(
                                        K_xn, 1, 2))).reshape((n_theta, n, 1)))
                        del dK_nn_inv_dtheta, dK_xn_dtheta, K_nn_inv, L, K_xn, K, K_gradient  # Free some memory, especially for GPU

                        log_likelihood_gradient = (-1 / sigma + (y_train - mu) ** 2 / sigma ** 3) * dsigma_dtheta + (
                                    y_train - mu) / sigma ** 2 * dmu_dtheta
                        del dmu_dtheta, dsigma_dtheta, sigma, mu, y_train  # Free some memory, especially for GPU

                        gradients.append(xp.to_numpy(xp.sum(
                            log_likelihood_gradient, axis=(1, 2))))  # Axis 0 is the theta parameter, axis 2 is the dimension of the output
    
    def sample_y(self, X, n_samples=1, random_state=0, conditioning_method=None):
        """Draw samples from Gaussian process and evaluate at X.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Query points where the GP is evaluated.

        n_samples : int, default=1
            Number of samples drawn from the Gaussian process per query point.

        random_state : int, np.random.RandomState instance or None, default=0
            Determines random number generation to randomly draw samples.
            Pass an int for reproducible results across multiple function
            calls.

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
        main_worker = self.workers[0]

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
                y_train = y_train[:, None]
            y_dim = y_train.shape[-1]
            kernel = self.kernel_

        # If conditioning_method is only-train, then each sample is independent of the others and we can use
        # a faster (full parallel) algorithm
        if conditioning_method == 'only-train':
            mu, sigma = self.predict(X, return_std=True)
            if y_dim == 1:
                mu, sigma = mu[..., None], sigma[..., None]  # Add y-dim dimension
            mu, sigma = mu[..., None], sigma[..., None]  # Add n_samples dimension
            assert mu.shape == sigma.shape
            y_samples = self._distribute_random_normal(random_state, mu, sigma, n_samples)

            # y_samples = y_samples.reshape((*shape, n_samples))
            if y_dim == 1:
                y_samples = y_samples[:, 0, :]  # Remove dimension corresponding to the y-dimension

            for worker in self.workers:
                worker.clear_cache()

            return y_samples
        
        # If conditioning_method is not only-train, continue here with the sequential algorithm
        xp = main_worker.get_array_module()
        rng = main_worker.check_random_state(random_state)

        # Find nearest neighbours. They could be searched in batches, but not really needed since the memory footprint is 
        # negligible. The biggest memory usage comes from the kernel K and its Cholesky decomposition.
        nn_indices_worker, x_full = self._find_nn_indices_for_train_and_eval(
            main_worker, kernel, x_train, X, condition_on_eval=conditioning_method != 'only-train',
            double_nn=conditioning_method == 'full-double-nn')
        nn_indices = xp.to_numpy(nn_indices_worker)

        # Allocate output and temporary vars
        y_samples = np.ones((nq, y_dim, n_samples)) * np.nan
        y_nn = np.empty((nn_indices.shape[1] - 1, y_dim, n_samples))

        # Loop over batches of data in case the entire arrays cannot be all stored in memory
        for i_batch in range(int(np.ceil(nq / self.batch_size))):
            i0 = self.batch_size * i_batch
            i1 = min(self.batch_size * (i_batch + 1), nq)

            # Evaluates the kernel and kernel gradient
            K, _ = main_worker.fill_nn_kernel(
                x_full, nn_indices_worker[i0:i1], kernel, eval_gradient=False)

            # Add jitter to the kernel
            ind = xp.where((nn_indices_worker[i0:i1] < nt) & (nn_indices_worker[i0:i1] >= 0))
            K[ind[0], ind[1], ind[1]] += self.alpha
            del ind

            # Calculate the Cholesky decomposition
            L, L_lower = main_worker.batched_chofactor(K[:, 1:, 1:], may_overwrite_x=False)
            # Use float 64 since on cpu doesn't give any significant gain to use float32. Memory usage is limited here.
            L, K = xp.to_numpy(L), xp.to_numpy(K)

            # Fill output
            for i in range(L.shape[0]):
                assert nn_indices[i + i0, 0] == nt + i0 + i
                this_ind = nn_indices[i + i0, 1:]
                is_neg = this_ind < 0
                is_train = (this_ind < nt) & (this_ind >= 0)
                not_train = this_ind >= nt
                non_train_ind = this_ind[not_train] - nt

                y_nn[is_neg, :, :] = 0
                y_nn[is_train, :, :] = y_train[this_ind[is_train]][:, :, None]
                y_nn[not_train, :, :] = y_samples[non_train_ind]

                this_K_xn = K[i, 0, 1:].reshape((1, -1))
                this_K_xn[0, is_neg] = 0
                this_K_nn_inv = lambda right: sl.cho_solve((L[i], L_lower), right, overwrite_b=False)

                if this_K_xn.size > 0:
                    mu = np.einsum('i,ijk->jk', this_K_nn_inv(this_K_xn.T).reshape((-1)),
                                   y_nn)  # k is the sample index, j is the y-dimension index, i is the nn index
                    sigma = max(0, np.sqrt(K[i, 0, 0] - (this_K_xn @ this_K_nn_inv(this_K_xn.T))))  # May be negative due to rounding
                else:
                    mu = 0
                    sigma = np.sqrt(K[i, 0, 0])

                y_samples[i + i0, :, :] = xp.to_numpy(main_worker.random_normal(rng, mu, sigma, (y_dim, n_samples)))

        if hasattr(self, '_y_train_std'):
            y_samples = y_samples * self._y_train_std.reshape((1, -1, 1)) + self._y_train_mean.reshape((1, -1, 1))  # Undo y scaling

        if y_dim == 1:
            y_samples = y_samples[:, 0, :]  # Remove dimension corresponding to the y-dimension

        for worker in self.workers:
                worker.clear_cache()

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

        main_worker = self.workers[0]
        xp = main_worker.get_array_module()

        # Faster calculation for prior
        if is_prior and not return_cov:
            mean = np.zeros(X.shape[0])
            if return_std:
                std = np.sqrt(xp.to_numpy(kernel.diag(xp, xp.from_numpy(X))))  # sqrt is intentionally in numpy
                return mean, std
            main_worker.clear_cache()
            return mean

        # Support multi-dimensional output of y_train
        if y_train.ndim == 1:
            y_train = y_train[:, None]
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
        
        # Find nearest neighbours.
        nn_indices, x_full = self._find_nn_indices_for_train_and_eval(
            main_worker, kernel, x_train, X, condition_on_eval=conditioning_method != 'only-train',
            double_nn=conditioning_method == 'full-double-nn')

        # If conditioning_method is only-train then each sample is independent of the others. Thus, if return_cov is False, we can use
        # a faster (full parallel) algorithm
        if (conditioning_method == 'only-train') and (not return_cov):
            nn_indices, x_full = xp.to_numpy(nn_indices), xp.to_numpy(x_full)

            # Downcast to float32 if needed
            if self.allow_downcast_f32 == "yes":
                x_full, y_train = x_full.astype('float32', copy=False), y_train.astype('float32', copy=False)

            # Run in parallel
            num_batches = int(np.ceil(nq / self.batch_size))
            batch_results = self.distribute(
                self._predict_worker,
                num_batches,
                common_arguments=(nq, nt, x_full, y_train, kernel, nn_indices, return_std))
            indexes = list(itertools.chain.from_iterable([x[0] for x in batch_results]))
            mean = list(itertools.chain.from_iterable([x[1] for x in batch_results]))
            sorter = np.argsort(indexes)
            mean = np.concatenate([mean[s] for s in sorter], axis=0)

            for worker in self.workers:
                worker.clear_cache()

            # Return output
            mean = format_mean(mean)
            if return_std:
                std = list(itertools.chain.from_iterable([x[2] for x in batch_results]))
                std = np.concatenate([std[s] for s in sorter], axis=0)
                return mean, format_sigma(std)
            return mean

        # If we reach this point, we have to go through the slow, sequential algorithm
        num_nn = nn_indices.shape[1]
        if num_nn > 1:
            nn_indices[:, 1:] = xp.sort(nn_indices[:, 1:], axis=1)  # To access partial covariance elements less randomly
        nn_indices_np = xp.to_numpy(nn_indices)

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
            col_ind = nn_indices_np[:, 1:].reshape(-1) - nt
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
            K, _ = main_worker.fill_nn_kernel(
                x_full, nn_indices[i0:i1], kernel, eval_gradient=False)

            # Add jitter to the kernel
            ind = xp.where((nn_indices[i0:i1] < nt) & (nn_indices[i0:i1] >= 0))
            K[ind[0], ind[1], ind[1]] += self.alpha
            del ind

            # Calculate the Cholesky decomposition
            L, L_lower = main_worker.batched_chofactor(K[:, 1:, 1:], may_overwrite_x=False)
            K, L = xp.to_numpy(K), xp.to_numpy(L)

            # Fill output
            for i in range(L.shape[0]):
                i_full = i + i0
                this_ind = nn_indices_np[i_full]
                assert this_ind[0] == nt + i_full
                this_ind = this_ind[1:]
                valid_ind_mask = this_ind > -1
                valid_ind = this_ind[valid_ind_mask]
                this_y[valid_ind_mask] = mean[valid_ind]
                this_y[~valid_ind_mask] = 0

                this_K_xn = K[i, 0, 1:].reshape((1, -1))
                this_K_nn_inv = lambda right: sl.cho_solve((L[i], L_lower), right, overwrite_b=False)

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
                partial_cov = partial_cov[:, :, None] * self._y_train_std.reshape((1, 1, -1)) ** 2
                if partial_cov.shape[2] == 1:
                    partial_cov = partial_cov[:, :, 0]
            return mean, partial_cov
        
        for worker in self.workers:
            worker.clear_cache()

        return mean

    def _predict_worker(self, i_worker, queue, nq, nt, x_full, y_train, kernel, nn_indices_np, return_std):
        """ Predicts the mean and standard deviation of the Gaussian Process on a given dataset conditioning only on the training dataset.
        Prediction is executed on a specific worker and only part of the given dataset is processed. Every time the worker has finished 
        processing a chunk/batch of data, a new batch is retrieved from the queue until all tasks have been processed.

        Parameters
        ----------
        i_worker : int
            Index of the worker to use

        queue : Queue like instance
            Queue used to retrieve the tasks to be performed

        nt : int
            Number of query points (i.e. points where mean and std should be evaluated)

        nt : int
            Number of training points

        x_full : np.ndarray
            Dataset consisting on the concatenation of the training and evaluation (query) datasets

        y_train : np.ndarray
            Realizations of the train dataset

        kernel : nngpr.batched_kernels.BatchedKernel
            Kernel to use

        nn_indices_np : np.ndarray of shape (nq, self.num_nn + 1)
            Nearest neighbors indices for the query points, as returned by self._find_nn_indices_for_train_and_eval.

        Returns
        -------
        indexes : list[int]
            List of indexes of the tasks processed by this worker

        means : list[np.ndarray]
            Gaussian Process means on the query points. means[i][j] is the mean of the j-th query point for the i-th processed
            task

        stds : list[np.ndarray]
            Gaussian Process standard deviations on the query points. stds[i][j] is the standard deviations of the j-th query 
            point for the i-th processed task
        """
        worker = self.workers[i_worker]
        worker_bs = worker.batch_size_for_nn(self.num_nn, x_full.itemsize, kernel)
        xp = worker.get_array_module()
        batch_size = self.batch_size #if worker.max_batch_size_per_nn is None else worker.max_batch_size_per_nn
        y_train = xp.from_numpy(y_train)

        indexes = []
        mus = []
        sigmas = []

        while True:
            try:
                index = queue.get_nowait()
                i0_main = index * batch_size
            except Empty:
                worker.clear_cache()
                return indexes, mus, sigmas
            
            i1_main = min(i0_main + batch_size, nq)
            local_batch_mu, local_batch_std = [], []

            # Continue the work in smaller chunks to prevent out of memory in the worker
            for i0 in range(i0_main, i1_main, worker_bs):
                i1 = min(i0 + worker_bs, i1_main)
            
                nn_indices = xp.from_numpy(nn_indices_np[i0:i1])
                # Evaluates the kernel and kernel gradient
                K, _ = worker.fill_nn_kernel(
                    xp.from_numpy(x_full), nn_indices, kernel, eval_gradient=False)
                # Add jitter to the kernel
                ind = xp.where((nn_indices < nt) & (nn_indices >= 0))
                K[ind[0], ind[1], ind[1]] += self.alpha
                del ind

                # Calculate y_train_nn
                y_train_nn = self._build_ytrain_given_nn_indices(worker, y_train, nn_indices[:, 1:])

                # Calculate the Cholesky decomposition
                L, L_lower = worker.batched_chofactor(K[:, 1:, 1:], may_overwrite_x=False)

                # Define relevant matrices
                K_xn = K[:, :1, 1:]
                def K_nn_inv(right):
                    return worker.batched_chosolve(L, right, L_lower)
                
                # Calculate mean
                mean = (K_xn @ K_nn_inv(y_train_nn))[:, 0, :]

                if return_std:
                    n = i1 - i0
                    std = xp.sqrt(K[:, 0, 0].reshape((n, 1)) - (K_xn @ K_nn_inv(xp.swapaxes(K_xn, 1, 2))
                                                            ).reshape((n, 1))).reshape(-1)
                    local_batch_std.append(xp.to_numpy(std))

                local_batch_mu.append(xp.to_numpy(mean))

            mus.append(np.concatenate(local_batch_mu, axis=0))
            if return_std:
                sigmas.append(np.concatenate(local_batch_std, axis=0))
            indexes.append(index)

    def get_default_kernel(self):
        """Returns the default kernel to use when no kernel is specified by the user

        Parameters
        ----------

        Returns
        -------

        kernel: batched_kernel.BatchedKernel
            Default kernel

        """
        return batched_kernels.ConstantKernel(1.0, constant_value_bounds="fixed") * batched_kernels.RBF(
            1.0, length_scale_bounds="fixed")

    def _find_nearest_neighbors_wrapper(self, worker, ref, query, *args, **kwargs):
        """ Wrapper function that calls worker._find_nearest_neighbors_wrapper, downcasting float64 to float32 if needed

        Parameters
        ----------
        worker : NngprWorker
            Worker

        ref : array
            Reference point array

        query : array
            Query point array

        Returns
        -------
        same arguments returned by NngprWorker.find_nearest_neighbors
        """
        xp = worker.get_array_module()
        if self.allow_downcast_f32 != 'no':
            ref, query = xp.to_float32(ref), xp.to_float32(query)
            
        return worker.find_nearest_neighbors(ref, query, self.nn_type, *args, **kwargs)

    def _distribute_random_normal(self, seed, mu, sigma, num_samples):
        """ Distributes sampling from the random normal distribution
        
        Parameters
        ----------
        seed : None, int or instance of np.random.RandomState
            Random seed to use for sampling.

        mu : np.ndarray of shape (*base_shape, 1)
            Mean of the normal distribution.

        sigma : np.ndarray of shape (*base_shape, 1)
            Standard deviation of the normal distribution. The shape must be the same as the shape of mu.

        num_samples : int
            Number of samples to draw

        Returns
        -------
        np.ndarray(Any) : Samples from the normal distribution, with shape (*base_shape, num_samples)

        """
        assert mu.shape == sigma.shape, "Mu and sigma sizes do not match"

        new_shape = (*mu.shape[:-1], num_samples)
        indexes, aggregator = self._split_random_generation(new_shape, mu.itemsize, self.random_gen_batch_byte_size)

        # Seed
        rng = check_random_state(seed)
        seeds = [int(x) for x in rng.randint(low=0, high=2**31, size=len(self.workers))]

        res = self.distribute(
            self._random_normal_wrapper, 
            len(indexes),
            worker_arguments=[[x] for x in seeds],
            common_arguments=(indexes, mu, sigma)
        )

        return aggregator(list(itertools.chain.from_iterable(res)))
    
    def _random_normal_wrapper(self, i_worker, queue, seed, indexes, mu, sigma):
        """ Helper function to distribute the random generation of normal distribution. 
        
        Parameters
        ----------
        i_worker : int
            Index of the worker to use

        queue : Queue-like
            Queue where tasks index are sourced from.

        seed : int
            Seed to use for the random generation

        indexes : list[tuple]
            List of tuples, where each tuple contains data specific for a task. In different words, when
            queue.get() returns task index i, then indexes[i] should be retrieved and used for the random
            generation task. Eaci list element (tuple) is made of three indexes: i0, i1 and n_samples. 
            Given i0, i1 and n_samples, the task is expected to generate n_samples random samples using 
            mu[i0:i1, ...] and sigma[i0:i1, ...]. The generated samples must have shape equal to 
            (i1-i0, ..., n_samples).

        mu : np.ndarray of shape (*base_shape, 1)
            Mean of the normal distribution.

        sigma : np.ndarray of shape (*base_shape, 1)
            Standard deviation of the normal distribution. The shape must be the same as the shape of mu.

        Returns
        -------
        list[np.ndarray] : List of samples generated by each task.

        """
        
        worker = self.workers[i_worker]
        xp = worker.get_array_module()
        rng = worker.check_random_state(seed)

        out = []
        while True:
            try:
                n = queue.get_nowait()
                i0, i1, n_samples = indexes[n]
            except Empty:
                worker.clear_cache()
                return out
            
            if worker.random_gen_batch_byte_size is None:
                local_indexes = [(i0, i1, n_samples)]
                aggregator = lambda x: x[0]
            else:
                local_indexes, aggregator = self._split_random_generation((i1-i0, *mu.shape[1:-1], n_samples), mu.itemsize, self.random_gen_batch_byte_size)

            local_out = []
            for i0, i1, n_samples in local_indexes:
                tmp_mu = mu[i0:i1, ...]
                local_out.append(xp.to_numpy(worker.random_normal(rng, tmp_mu, sigma[i0:i1, ...], shape=(*tmp_mu.shape[:-1], n_samples))))
            out.append(aggregator(local_out))
    
    @staticmethod
    def _split_random_generation(shape, itemsize, batch_byte_size):
        """ Helper function that splits a random generation request into batches of maximum size batch_byte_size bytes.

        Parameters
        ----------
        shape : tuple[int]
            Shape of the array to be generated.

        itemsize : int
            Size in bytes of each element of the array to be generated.

        batch_byte_size : int
            Maximum number of bytes that can be allocated for a single batch.

        Returns
        -------
        indexes : list[tuple[int]]
            List of tuples, where each tuple contains the start index (i0), end index (i1) and number of samples 
            (n_samples) for each batch. In different words, each batch should generate random samples of shape 
            (i1-i0, ..., n_samples), where ellipsis ... complete the missing dimensions as from shape input parameter.

        aggregator : callable[list[array]] -> array
            Aggregator function that takes a list of arrays generated by each batch and returns an array with the same 
            shape as the 'shape' input parameters to this function. This is used to concatenate the batches into one array.
        
        """
        assert len(shape) > 1

        num_samples = shape[-1]

        # Batch size if we parallelize over num_samples
        batch_size_y = max(1, batch_byte_size // (itemsize * np.prod(shape[:-1])))
        n_batch_y = max(1, num_samples // batch_size_y)

        # Batch size if we parallelize over mu.shape[0]
        batch_size_x = max(1, batch_byte_size // (itemsize * np.prod(shape[1:])))
        n_batch_x = max(1, shape[0] // batch_size_x)

        par_over_y = n_batch_y > n_batch_x

        if par_over_y:
            bs = np.diff(np.minimum(np.concatenate([[0], np.cumsum(np.ones(n_batch_y) * batch_size_y)]), num_samples))
            bs = bs[bs > 0]
            assert np.sum(bs) == num_samples
            indexes = [(0, shape[0], int(x)) for x in bs]
        else:
            bs = np.diff(np.minimum(np.concatenate([[0], np.cumsum(np.ones(n_batch_x) * batch_size_x)]), shape[0]))
            bs = bs[bs > 0]
            assert np.sum(bs) == shape[0]
            bs = np.cumsum(bs)
            bs_left = np.concatenate([[0], bs[:-1]])
            bs_right = bs
            indexes = [(int(l), int(r), num_samples) for l, r in zip(bs_left, bs_right)]

        def aggregator(res):
            return np.concatenate(res, axis=-1 if par_over_y else 0)
        
        return indexes, aggregator
    
    @staticmethod
    def _build_ytrain_given_nn_indices(worker, y_train, nn_indices):
        """Calculates the array containing the observed y values for each nearest neighbor.

        Parameters
        ----------

        worker : NngprWorker
            Worker to be used for the calculation.

        y_train : array of shape (n_train,)
            Observed y values in the training dataset.

        nn_indices : array of shape (n_query, n_nn)
            Array that at position [i, j] stores the index of the j-th nearest neighbor for the i-th query point. nn_indices[i, j]
            must be set to -1 when the j-th nearest neighbor for the i-th query point is not defined.

        Returns
        -------

        y_train_nn : array of the same shape (n_query, n_nn)
            Array that at position [i, j] stores the observed y_train on the j-th nearest neighbor for the i-th query point. If the j-th
            nearest neighbor index in nn_indices is -1, then y_train_nn[i, j] is set to zero.

        """

        xp = worker.get_array_module()
        y_train_nn = xp.empty((*nn_indices.shape, y_train.shape[-1]), dtype=y_train.dtype)
        usable = nn_indices != -1
        y_train_nn[usable, :] = y_train[xp.to_numpy(nn_indices[usable]), :]  # TODO check
        y_train_nn[~usable, :] = 0

        return y_train_nn

    def _find_nn_indices_for_train_and_eval(self, worker, kernel, x_train, x_query, condition_on_eval, double_nn):
        """Calculates the array containing the observed y values for each nearest neighbor.

        Parameters
        ----------

        worker : NngprWorker
            Worker to be used for the calculation.

        kernel : batched_kernels.BatchedKernel
            Kernel of the gaussian process

        x_train : array of shape (n_train, n_features)
            Array containing the training points

        x_query : array of shape (n_query, n_features)
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

        nn_indices : array of shape (n_query, n_nn + 1) or (n_query, n_nn*2 + 1)
            Array that at position [i, j], when j > 0, stores the index of the j-th nearest neighbor for the i-th query point.
            nn_indices[i, 0] stores the index of the i-th query point. Indices are meant to be used to lookup points inside the `x_full`
            array, which is returned by this function as second argument. When the j-th nearest neighbor for the i-th query point does not
            exist, then nn_indices[i, j] is set to -1.

        x_full : array of shape (n, n_features)
            Array that can be used to lookup points given the indices stored in nn_indices. It's either equal to `x_train` or the
            concatenation of `x_train` and `x_query` depending on the input variables.

        """

        xp = worker.get_array_module()
        x_train, x_query = xp.from_numpy(x_train), xp.from_numpy(x_query)

        def get_x_full():
            return xp.concatenate([x_train, x_query])

        x_full = None
        if condition_on_eval:
            if double_nn:
                if x_train.shape[0] > 0:
                    nn_ind_train = self._find_nearest_neighbors_wrapper(
                        worker, x_train, x_query, kernel, min(self.num_nn, x_train.shape[0]), ref_cut_index=None)
                else:
                    nn_ind_train = xp.empty((0, 0), dtype=np.int32)
                nn_ind_nontrain = self._find_nearest_neighbors_wrapper(
                    worker, x_query, x_query, kernel, min(self.num_nn, x_query.shape[0] - 1), ref_cut_index=0)
                nn_ind_nontrain[nn_ind_nontrain != -1] += x_train.shape[0]
                if x_train.shape[0] > 0:
                    # Need to combine nn_ind_train and nn_ind_nontrain
                    nn_indices = xp.empty((nn_ind_train.shape[0], nn_ind_train.shape[1] + nn_ind_nontrain.shape[1]),
                                          dtype=nn_ind_train.dtype)
                    arange = xp.tile(xp.arange(nn_indices.shape[1], dtype=np.int32), (nn_indices.shape[0], 1))

                    # First insert indices from indices_0
                    is_valid = nn_ind_train != -1
                    n_to_add_0 = xp.sum(is_valid, axis=1).reshape((-1, 1))
                    nn_indices[arange < n_to_add_0] = nn_ind_train[is_valid]

                    # Then insert indices from indices_1
                    is_valid = nn_ind_nontrain != -1
                    n_to_add_1 = n_to_add_0 + xp.sum(is_valid, axis=1).reshape((-1, 1))
                    nn_indices[(arange >= n_to_add_0) & (arange < n_to_add_1)] = nn_ind_nontrain[is_valid]

                    # Set the rest to the null value
                    nn_indices[arange >= n_to_add_1] = -1
                else:
                    nn_indices = nn_ind_nontrain
                del nn_ind_train, nn_ind_nontrain
            else:
                x_full = get_x_full()
                nn_indices = self._find_nearest_neighbors_wrapper(
                    worker, x_full, x_query, kernel, min(self.num_nn, x_full.shape[0]), ref_cut_index=x_train.shape[0])
        else:
            num_nn = min(x_train.shape[0], self.num_nn)
            if num_nn > 0:
                nn_indices = self._find_nearest_neighbors_wrapper(worker, x_train, x_query, kernel, num_nn, ref_cut_index=None)
            else:
                nn_indices = xp.empty((x_query.shape[0], 0), dtype=np.int32)

        nn_indices = xp.concatenate([
            xp.arange(x_train.shape[0], x_train.shape[0] + nn_indices.shape[0]).reshape((-1, 1)), nn_indices], axis=1)
        x_full = get_x_full() if x_full is None else x_full
        return nn_indices, x_full


class ArrayModule(ABC):
    """ Interface for module that should mimick numpy. The Array module must implemnt at least the following numpy functions
    in additions to the abstract methods: 
        moveaxis, swapaxes, log, sqrt, arange, where, sum, tile, asarray.
    """

    @abstractmethod
    def to_numpy_impl(self, x):
        """ Converts the input array to a numpy ndarray
        
        Parameters
        ----------
        array : array of array module type
            Input array

        Returns
        -------
        array : np.ndarray
        """
        pass

    @abstractmethod
    def from_numpy(self, x):
        """ Converts the input numpy array to an array-module array type
        
        Parameters
        ----------
        array : np.ndarray
            Input array

        Returns
        -------
        array : array of array module type
        """
        pass

    def to_numpy(self, x):
        """ Converts the input array to a numpy ndarray if it's not already a numpy ndarray
        
        Parameters
        ----------
        array : np.ndarray or array of array module type
            Input array

        Returns
        -------
        array : np.ndarray
        """
        if type(x) is np.ndarray:
            return x
        return self.to_numpy_impl(x)
    
    @abstractmethod
    def to_float32(self, x):
        """ Convert the input array to float32 dtype 
        
        Parameters
        ----------
        array : array of array module type
            Input array

        Returns
        -------
        array : input array in float 32 dtype
        """
        pass

    @abstractmethod
    def cdist(self, x, y):
        """ Calculates pairwise euclidean distance between points in x and y
        
        Parameters
        ----------

        x : array of shape (*base_shape, nx, nf)
            Left array for pairwise distance

        y : array of shape (*base_shape, ny, nf)
            Right array for pairwise distance

        Returns
        -------

        array of shape (*base_shape, nx, ny)
            Pairwise euclidean distance between points in x and y. Element at index (..., ix, iy) is the distance
            between x[..., ix, :] and y[..., iy, :]

        """
        pass


class NngprWorker(ABC):

    """ Base class for the creation of workers to be used with NumpyNngpr. Allows for easy integration of different Array modules,
    like Numpy, Cupy, Torch, Jax, etc


    Parameters
    ----------
    random_gen_batch_byte_size : int | None, optional, default None
        Specifies the maximum size in bytes for a random generated array. Must be an int or None (no size limit). When the size
        of a random sample exceeds this limit, requests to the worker are split in chunks.

    nn_large_size_preferred : int | None, optional, default None
        Specifies preferred batch size to be used for the neares neighbors search. Some algorithms may be faster when large batch
        sizes are used (e.g. the search implemented in cuda_kernel_accelerators), thus this variable specifies the preferred batch 
        size for it. Note that this batch size affects only the neared neighbors search, other operations are controlled by the 
        batch size returned by the method batch_size_for_nn.

    batch_size : int | None, optional, default None
        If not None, batch size to be returned by the method batch_size_for_nn. See also batch_size_for_nn docstrings.
    """

    def __init__(self, random_gen_batch_byte_size=None, nn_large_size_preferred=None, batch_size=None):
        if random_gen_batch_byte_size is not None:
            assert random_gen_batch_byte_size > 0
        self.random_gen_batch_byte_size = random_gen_batch_byte_size
        if nn_large_size_preferred is not None:
            assert nn_large_size_preferred > 0
        self.nn_large_size_preferred = nn_large_size_preferred
        if batch_size is not None:
            assert batch_size > 0
        self.batch_size = batch_size
        super().__init__()

    @abstractmethod
    def get_array_module(self):
        """ Function that returns the array module (Numpy-like) to be used with the worker.
        
        Returns
        -------

        array module: ArrayModule 
            Numpy-like array module. 
        
        """
        
        pass

    def clear_cache(self):
        """ This method clears the cache used by the worker. Implements in derived classes in case the worker uses some cache.
        For example, in case of Numpy-based workers no cache is used. For Cuda-base workes, often libraries keep some memory
        pools for faster allocation. Sometimes this needs to be cleared to allow for other applications to use GPU memory
        """
        pass

    def find_nearest_neighbors(self, ref, query, nn_type, kernel, num_nn, ref_cut_index):
        """Finds the nearest neighbors in 'ref' for each point in 'query'.

        Parameters
        ----------

        ref : array of shape (n_reference_points, n_features)
            Search points for the nearest neighbors search

        query : array of shape (n_query_points, n_features)
            Query points for the nerest neighbors search

        nn_type : str, default 'kernel-space'
            Search space for the nearest neighbors. Can be either 'kernel-space' or 'input-space'. If 'kernel-space' nearest neighbors
            are searched in the kernel space, i.e. the neighbors of a query point are the points with the highest covariance w.r.t. the 
            query point. When 'input-space' nearest neighbors are searched in the input feature space, using euclidean distance.

        kernel : batched_kernels.BatcheKernel
            Kernel of the gaussian process

        num_nn : int
            number of nearest  neighbors to use

        ref_cut_index : int
            If negative, the nearest neighbours for the j-th point in query are searched for among all points in ref.
            If positive, the nearest neighbours for the j-th point in query are searched for only among points in ref[:j + ref_cut_index].

        Returns
        -------
        nn_indices : array of shape (n_query_points, num_nn)
            Nearest neighbour indices. nn_indices[i, j] contains the index of the j-th nearest neighbor in `ref` of the i-th point in
            `query`. If the j-th nearest neighbor for the i-th query points does not exist (e.g. because the search space has less than
            j points, possibly due to the usage of non-negative `ref_cut_index`), then nn_indices[i, j] is set to -1.
        """

        xp = self.get_array_module()
        nn_indices = -xp.ones((query.shape[0], num_nn), dtype=np.int32)
        i_q_max = 0
        if ref_cut_index is not None:
            i_q_max = max(0, 1 - ref_cut_index)  # Query points before this index do not have any neighbors
            
        # Loop over mini-batches to reduce memory usage
        while i_q_max < query.shape[0]:
            i_q_min = i_q_max
            batch_size = max(1, int(1024**3 * 0.5 / (ref.itemsize * ref.shape[0])))  # Limit the batch size to exploit the ref_cut_index and limit the memory usage
            i_q_max = min(nn_indices.shape[0], i_q_min + batch_size)
            if ref_cut_index is not None:
                this_ref = ref[..., :i_q_max + ref_cut_index, :]  # Drop points that are useless
            else:
                this_ref = ref
                
            if nn_type == 'kernel-space':
                dists = -kernel(xp, query[i_q_min:i_q_max, :], this_ref)
            else:
                dists = xp.cdist(query[i_q_min:i_q_max, :], this_ref)
            if ref_cut_index is not None:
                unmask = (xp.arange(i_q_min, i_q_max, dtype=xp.int32)[:, None] + ref_cut_index) \
                    <= xp.arange(this_ref.shape[0], dtype=xp.int32)[None, :]
                dists[unmask] = xp.inf
            indices = xp.argsort(dists, -1)[..., :num_nn]
            
            if ref_cut_index is not None:
                if unmask.shape[1] >= indices.shape[1]:
                    indices[unmask[:, :indices.shape[1]]] = -1
                else:
                    unmask = (xp.arange(i_q_min, i_q_max, dtype=xp.int32)[:, None] + ref_cut_index) \
                        <= xp.arange(indices.shape[1], dtype=xp.int32)[None, :]
                    indices[unmask] = -1
                del unmask
                
            nn_indices[i_q_min:i_q_max, :indices.shape[1]] = indices
            del indices, dists, this_ref

        return nn_indices

    @abstractmethod
    def batched_chosolve(self, L, y, lower, may_overwerite_b=False):
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

        x : array of the same shape of input y
            Solution arrays X_i of A_i * X_i = B_i.

        """
        pass

    @abstractmethod
    def batched_chofactor(self, x, may_overwrite_x=False):
        """Calculates the Cholesky decomposition for a batch of symmetric and square matrices x_i.

        Parameters
        ----------

        x : array of shape (..., n, n)
            Batch of symmetric matrices for which the Cholesky decomposition is calculated.

        lower : bool
            If True, then the Cholesky decomposition is stored in the lower triangular part of L_i, else in the upper triangular part.
            Note that the triangular part which does not store the cholesky decomposition does not contain valid numbers (i.e. it may be
            not zeroed).

        overwrite_x : bool, default False
            If True, the Cholesky decomposition may be stored in the same input array x to avoid new memory allocation.

        Returns
        -------

        L : array of the same shape of input x
            Batch of Cholesky decomposition of input arrays x.

        """
        
        pass

    @abstractmethod
    def check_random_state(self, seed):
        """Turn seed into a random state (RS) instance. Tye type of RS depends on the specific child class.

        Parameters
        ----------
        seed : None, int or instance of random state (RS)

        Returns
        -------
        :class:RS
            The random state object based on `seed` parameter.
        """
        pass

    @abstractmethod
    def random_normal(self, random_state, mu, sigma, shape):
        """ Calculates samples from a random normal distribution.

        Parameters
        ----------
        random_state : Random State instance as returned by 'check_random_state'
            Random generator

        mu : array
            Mean of the normal distribution

        sigma : array
            Standard deviation of the normal distribution

        shape : tuple(int)
            Shape of the generated samples. Shape of mu and sigma must be broadcastable to this shape

        Returns
        -------
        :array
            Array of normally distributed random samples of shape 'shape' 
        """
        pass

    def fill_nn_kernel(self, x, nn_indices, kernel, eval_gradient=False):
        """Calculates the kernel based on nearest neighbors given the nearest neighbors indices.

        Parameters
        ----------
        x : array of shape (n_points, n_features)
            Input dataset

        nn_indices : array of shape (n_query, n_nn)
            Array that at position [i, j] stores the index of the j-th nearest neighbor for the i-th query point. nn_indices[i, j]
            must be set to -1 when the j-th nearest neighbor for the i-th query point is not defined.

        kernel : batched_kernels.BatcheKernel
            Kernel of the gaussian process

        Returns
        -------
        K : array of the same shape (n_query, n_nn, n_nn)
            K[i, j, k] is the kernel evaluated between the j-th and k-th nearest neighbors of the i-th query point.

        K_gradient : array of the same shape (n_query, n_nn, n_nn, n_theta)
            K[i, j, k, :] is the kernel gradient evaluated between the j-th and k-th nearest neighbors of the i-th query point. Returned
            only if 'eval_gradient' is set to True.

        """

        xp =self.get_array_module()
        # Evaluate kernel
        res = kernel(xp, x[nn_indices, :], eval_gradient=eval_gradient)
        if eval_gradient:
            nn_kernel, nn_kernel_grad = res
        else:
            nn_kernel = res
            nn_kernel_grad = xp.empty(0, dtype=x.dtype)

        # Zeroes the kernel and grad where necessary
        is_negative = xp.where(nn_indices == -1)
        nn_kernel[is_negative[0], is_negative[1], :] = 0
        nn_kernel[is_negative[0], :, is_negative[1]] = 0
        if eval_gradient:
            nn_kernel_grad[is_negative[0], is_negative[1], :] = 0
            nn_kernel_grad[is_negative[0], :, is_negative[1]] = 0
        
        nn_kernel[is_negative[0], is_negative[1], is_negative[1]] = 1
        del is_negative
        return nn_kernel, nn_kernel_grad

    def batch_size_for_nn(self, num_nn, itemsize, kernel):
        """ Returns the recommended batch size to be used by Nngpr in order to avoid out of memory errors on the worker. If
        self.batch_size is not None, self.batch_size is returned. Otherwise self.custom_batch_size_for_nn(...) will be
        returned.
        Specific implementations of NngprWorker are recommended to override custom_batch_size_for_nn, not batch_size_for_nn.
        
        Parameters
        ----------

        num_nn : int
            Number of nearest neighbors that will be used

        itemsize : int
            Size in bytes of each array element (e.g. 4 for float32, 8 for float64)

        kernel : nngpr.batched_kernels.BatchedKernel
            Kernel that will be used
        
        """
        if self.batch_size is None:
            return self.custom_batch_size_for_nn(num_nn, itemsize, kernel)
        return self.batch_size
    
    def custom_batch_size_for_nn(self, num_nn, itemsize, kernel):
        """ Returns the recommended batch size to be used by Nngpr in order to avoid out of memory errors on the worker. 
        
        Parameters
        ----------

        num_nn : int
            Number of nearest neighbors that will be used

        itemsize : int
            Size in bytes of each array element (e.g. 4 for float32, 8 for float64)

        kernel : nngpr.batched_kernels.BatchedKernel
            Kernel that will be used
        
        """
        return max(1, min(5000, int((1024**2 * 300) / (itemsize * max(num_nn, 128)**2))))


class DummyQueue(collections.deque):
    """ Class that extends the deque class with the 'get' and 'get_nowait' methods to mimick a Queue object """

    def get_nowait(self):
        if len(self) == 0:
            raise Empty()
        
        return self.popleft()

    def get(self, *args, **kwargs):
        return self.get_nowait()
