import numbers
import functools

import numpy as np

try:
    import cupy as cp
    from cupy.cuda import device
    from cupy_backends.cuda.libs import cublas
except ImportError:
    raise ImportError("This module requires cupy. Install with `pip install nngpr[cupy]`")

from .base_nngpr import Nngpr, NngprWorker, ArrayModule
from .cuda_kernel_accelerators import CudaKernelAccelerator


class CupyNngpr(Nngpr):

    """ This class is the Nngpr end-user interface that uses a Cupy-based backend.
    
    Parameters
    ----------

    kernel : see parent class docstring

    cuda_device_indexes: list[int] | None
        List of cuda device indexes to use. If None, all cuda devices will be used.

    alpha : see parent class docstring

    optimizer : see parent class docstring

    n_restarts_optimizer : see parent class docstring

    normalize_y : see parent class docstring

    copy_X_train : see parent class docstring

    n_targets : see parent class docstring

    random_state : see parent class docstring

    num_nn : see parent class docstring

    nn_type : see parent class docstring

    batch_size : see parent class docstring

    allow_downcast_f32 : see parent class docstring

    distribute_method : see parent class docstring

    mp_start_method : see parent class docstring

    random_gen_batch_byte_size : see parent class docstring

    distribute_environ : see parent class docstring

    """

    _parameter_constraints: dict = {
        **Nngpr._parameter_constraints,
        "cuda_device_indexes": [list]
    }

    def __init__(
            self,
            kernel=None,
            *,
            cuda_device_indexes=None,
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

        self.cuda_device_indexes = list(range(cp.cuda.runtime.getDeviceCount())) if cuda_device_indexes is None else cuda_device_indexes
        kernel = self.get_default_kernel() if kernel is None else kernel
        super().__init__(
            [CudaWorker(x, kernel) for x in self.cuda_device_indexes], num_nn=num_nn, kernel=kernel, alpha=alpha, optimizer=optimizer, 
            n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y, 
            copy_X_train=copy_X_train, random_state=random_state, batch_size=batch_size,
            nn_type=nn_type, allow_downcast_f32=allow_downcast_f32, distribute_method=distribute_method, mp_start_method=mp_start_method,
            random_gen_batch_byte_size=random_gen_batch_byte_size, distribute_environ=distribute_environ
            )


class CupyProxy(ArrayModule):

    """ Implements a Cupy-based array module. Thin wrapper around cupy which only adds support for the batched 
    cdist function and takes care of allocating arrays on the selected Cuda device. It also adds the numpy `asarray`
    interface to cupy.

    Parameters
    ----------

    cuda_device_index : int
        Index of the cuda device to use by this class instance

    """

    def __init__(self, cuda_device_index):
        self.cuda_device_index = cuda_device_index
        self.overrides_map = {
            'cdist': self.cdist,
            'asarray': self.asarray
        }

    def to_numpy_impl(self, x):
        """ Implements parent class abstract method. See parent class docstring """
        return x.get()

    def from_numpy(self, x):
        """ Implements parent class abstract method. See parent class docstring """
        with cp.cuda.Device(self.cuda_device_index):
            return cp.array(x)
        
    def to_float32(self, x):
        """ Implements parent class abstract method. See parent class docstring """
        return x.astype(np.float32, copy=False)

    def __getattr__(self, name):
        """ Exposes cupy attributes in this class """
        if (fn := self.overrides_map.get(name)) is not None:
            return fn
        if name in {'empty', 'zeros', 'ones', 'full', 'eye', 'arange'}:
            return functools.partial(self.redirect_with_device, name)
        return getattr(cp, name)
    
    def redirect_with_device(self, name, *args, **kwargs):
        """ Executes the cupy callable named `name` with arguments *args and **kwargs inside the GPU at index self.cuda_device_index """
        with cp.cuda.Device(self.cuda_device_index):
            return getattr(cp, name)(*args, **kwargs)

    def cdist(self, x, y):
        """ Calculates pairwise euclidean distance between points in x and y
        
        Parameters
        ----------

        x : cp.ndarray of shape (*base_shape, nx, nf)
            Left array for pairwise distance

        y : cp.ndarray of shape (*base_shape, ny, nf)
            Right array for pairwise distance

        Returns
        -------

        cp.ndarray of shape (*base_shape, nx, ny)
            Pairwise euclidean distance between points in x and y. Element at index (..., ix, iy) is the distance
            between x[..., ix, :] and y[..., iy, :]

        """
        return cdist_cuda(x, y)
    
    def asarray(self, x, copy=False, **kwargs):
        """ Adds numpy.asarray interface to cupy """
        with cp.cuda.Device(self.cuda_device_index):
            return cp.array(x, copy=copy, **kwargs)


class CudaWorker(NngprWorker):

    """ Implements NngprWorker with Cupy arrays 
    
    Parameters
    ----------

    cuda_device_index : int
        Index of the cuda device to use for the worker

    kernel : nngpr.batched_kernels.BatchedKernel
        Kernel that will be used with the worker. This kernel is only used in the constructor to determine whether cuda
        acceleration is implemented for this type of kernel. It is not further stored insi

    random_gen_batch_byte_size : see parent class docstring

    nn_large_size_preferred : see parent class docstring

    accelerate_kernel : bool | None
        Flag indicating whether the accelerated nearest neighbor search and kernel filling functions should be used.
        If True, the accelerated functions (from nngpr.cuda_kernel_accelerators) are used and an exception will be thrown
        if nngpr.cuda_kernel_accelerators cannot accelerate the given kernel. If False, the accelerated functions are
        not used. If None, accelerated functions are used only if the given kernel is supported.

    Attributes
    ----------
    total_memory_bytes : int
        Global memory in bytes of the cuda device in use

    cp : CupyProxy
        CupyProxy instance

    accelerate_kernel : bool | None
        Same as accelerate_kernel arguments

    kernel_acc : CudaKernelAccelerator
        CudaKernelAccelerator instance

    """

    def __init__(self, cuda_device_index, kernel, random_gen_batch_byte_size=None, nn_large_size_preferred=None, 
                 accelerate_kernel=None, **super_kwargs):
        
        total_memory_bytes = cp.cuda.runtime.getDeviceProperties(cuda_device_index)['totalGlobalMem']
        random_gen_batch_byte_size = total_memory_bytes // 4 if random_gen_batch_byte_size is None else random_gen_batch_byte_size
        self.total_memory_bytes = total_memory_bytes
        self.cp = CupyProxy(cuda_device_index)
        self.accelerate_kernel = accelerate_kernel
        self.kernel_acc = CudaKernelAccelerator()
        if (nn_large_size_preferred is None) and (self.accelerate_kernel is not False) and (self.kernel_acc.can_accelerate_kernel(kernel)):
            nn_large_size_preferred = 25000  # Accelerated nearest neighbor kernel works faster with large batches. This number is
                                             # based on heuristic
        super().__init__(random_gen_batch_byte_size=random_gen_batch_byte_size, 
                         nn_large_size_preferred=nn_large_size_preferred, **super_kwargs)
        
    def get_array_module(self):
        """ Implements parent class abstract method. See parent class docstring """
        return self.cp

    def clear_cache(self):
        """ Implements parent class abstract method. See parent class docstring """
        self._free_cp_memory()

    def batched_chosolve(self, L, y, lower, may_overwerite_b=False):
        """ Implements parent class abstract method. See parent class docstring """
        
        assert len(L.shape) > 2
        assert L.shape[-2] == L.shape[-1], "Not square matrices"
        assert len(y.shape) == len(L.shape), "y does not have the same number of dimensions as L"

        with cp.cuda.Device(self.cp.cuda_device_index):
            shape = cp.broadcast_shapes(L.shape[:-2], y.shape[:-2])
            copied = False
            if shape != L.shape[:-2]:
                new_l = cp.empty((*shape, *L.shape[-2:]), dtype=L.dtype, 
                                 order='F' if L.flags['F_CONTIGUOUS'] else 'C')
                new_l[...] = L
                L = new_l
            if shape != y.shape[:-2]:
                new_y = cp.empty((*shape, *y.shape[-2:]), dtype=y.dtype, 
                                 order='F' if y.flags['F_CONTIGUOUS'] else 'C')
                new_y[...] = y
                y = new_y
                copied = True

            if not may_overwerite_b and not copied:
                y = cp.array(y)  # We need to make a copy to avoid overwriting

            return cusolver_potrs(L, y, lower)

    def batched_chofactor(self, x, may_overwrite_x=False):
        """ Implements parent class abstract method. See parent class docstring """
        with cp.cuda.Device(self.cp.cuda_device_index):
            return cp.linalg.cholesky(x), True
    
    def check_random_state(self, seed):
        """Turn seed into a cp.random.Generator instance.

        Parameters
        ----------
        seed : None, int or instance of cp.random.Generator
            If seed is None, returns cp.random.default_rng()
            If seed is an int, returns  cp.random.default_rng(seed)
            If seed is already an instance cp.random.Generator, the same instance is returned
            Otherwise raise ValueError.

        Returns
        -------
        :class:`cp.random.Generator
            The random generator object based on `seed` parameter.

        """
        return check_random_state(seed)

    def random_normal(self, random_state, mu, sigma, shape):
        """ Implements parent class abstract method. See parent class docstring """
        with cp.cuda.Device(self.cp.cuda_device_index):
            mu, sigma = cp.array(mu), cp.array(sigma)
            return random_state.standard_normal(size=shape, dtype=mu.dtype) * sigma + mu

    def _free_cp_memory(self):
        """ Helper functions that frees some allocated but unused memory by cupy """
        with cp.cuda.Device(self.cp.cuda_device_index):
            cp.get_default_memory_pool().free_all_blocks()

    def _accelerate_if_needed(self, kernel, base_fn, acc_fn):
        """ Helper function that calls either the base function or the accelerated ones depending on whether acceleration 
        is supported for the given kernel 
        
        Parameters
        ----------

        kernel : batched_kernels.BachedKernel
            Kernel fo the gaussian process

        base_fn : callable
            Base function (i.e. slower than the accelerated one).

        acc_fn : callable
            Accelerated function (i.e. faster than the accelerated one).
        
        Returns
        -------

        Any: values returned by the base or accelerated function

        """
        if (self.accelerate_kernel is False) or ((self.accelerate_kernel is None) and (not self.kernel_acc.can_accelerate_kernel(kernel))):
            return base_fn()
        
        if (self.accelerate_kernel is True) and (not self.kernel_acc.can_accelerate_kernel(kernel)):
            raise RuntimeError("Cannot accelerate this kernel")
        
        return acc_fn()

    def find_nearest_neighbors(self, ref, query, nn_type, kernel, num_nn, ref_cut_index):
        """ Overrides parent class method. See parent class docstring """
        base_fn = functools.partial(
            super().find_nearest_neighbors, ref, query, nn_type, kernel, num_nn, ref_cut_index)
        
        acc_fn = functools.partial(
            self.kernel_acc.find_nearest_neighbors, self.cp.cuda_device_index, ref, query, nn_type, kernel, num_nn, ref_cut_index)

        # Note: the accelerate function is slower than the base one when the number of query points is less that roughly 20000. However,
        # the base function has some memory-usage issues (due to Cupy) caused apparently by not releasing unused memory and may result
        # in out of memory errors. For this reason, the accelerated function is preferred.
        
        return self._accelerate_if_needed(kernel, base_fn, acc_fn)
    
    def fill_nn_kernel(self, x, nn_indices, kernel, eval_gradient=False):
        """ Overrides parent class method. See parent class docstring """
        base_fn = functools.partial(
            super().fill_nn_kernel, x, nn_indices, kernel, eval_gradient=eval_gradient)
        acc_fn = functools.partial(
            self.kernel_acc.fill_nn_kernel, self.cp.cuda_device_index, x, nn_indices, kernel, eval_gradient=eval_gradient)
        
        return self._accelerate_if_needed(kernel, base_fn, acc_fn)
    
    def custom_batch_size_for_nn(self, num_nn, itemsize, kernel):
        """ Implements parent class abstract method. See parent class docstring """

        # Guess a size
        if (self.accelerate_kernel is not False) and self.kernel_acc.can_accelerate_kernel(kernel):
            max_batch_size = self.nn_large_size_preferred
        else:
            max_batch_size = self.total_memory_bytes * 2.5e-7

        # Part 2: limit based on number of nearest neighbors
        max_batch_size = min(max_batch_size, self.total_memory_bytes / (20 * itemsize * num_nn**2))
        
        # Part 3: limit based on gradient size
        max_batch_size = min(max_batch_size, self.total_memory_bytes / (5 * num_nn**2 * max(1, len(kernel.theta)) * itemsize) )

        return max(1, int(max_batch_size))    


def check_random_state(seed):
    """Turn seed into a cp.random.Generator instance.

    Parameters
    ----------
    seed : None, int or instance of cp.random.Generator
        If seed is None, returns cp.random.default_rng()
        If seed is an int, returns  cp.random.default_rng(seed)
        If seed is already an instance cp.random.Generator, the same instance is returned
        Otherwise raise ValueError.

    Returns
    -------
    :class:`cp.random.Generator
        The random generator object based on `seed` parameter.

    """
    if (seed is None) or isinstance(seed, numbers.Integral):
        return cp.random.default_rng(seed)
    # In this case random_state must be a valid cp.random.Generator
    if not isinstance(seed, cp.random.Generator):
        raise ValueError("Seed is not a cp.random.Generator instance or an integer")
    return seed


def cusolver_potrs(L, b, lower):
    """ Implements lapack XPOTRS through cusolver.potrs. Solves linear system
    A x = b given the cholesky decomposition of A, namely L. Supports also
    batches of linear systems and more than one right-hand side (NRHS > 1).

    Parameters
    ----------
    L (cupy.ndarray): Array of Cholesky decomposition of real symmetric or
        complex hermitian matrices with dimension (..., N, N).
    b (cupy.ndarray): right-hand side (..., N) or (..., N, NRHS). Note that
        this array may be modified in place, as usually done in LAPACK
    lower (bool): If True, L is lower triangular. If False, L is upper
        triangular.

    Returns
    -------
        cupy.ndarray: The solution to the linear system. Note this may point to
            the same memory as b, since b may be modified in place.
    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.
    """

    from cupy_backends.cuda.libs import cusolver as _cusolver

    # Check if batched should be used
    if len(L.shape) > 2:
        return cusolver_potrs_batched(L, b, lower)

    # Check input arguments
    assert L.ndim == 2, "L is not a matrix"
    assert L.shape[0] == L.shape[1], "L is not a square matrix"
    n = L.shape[-1]
    b_shape = b.shape
    if b.ndim == 1:
        b = b[:, None]
    assert b.ndim == 2, "b is not a vector or a matrix"
    assert b.shape[0] == n, "length of arrays in b does not match size of L"

    # Check memory order and type
    dtype = np.promote_types(L.dtype, b.dtype)
    dtype = np.promote_types(dtype, 'f')
    L, b = L.astype(dtype, copy=False), b.astype(dtype, copy=False)
    if (not L.flags.f_contiguous) and (not L.flags.c_contiguous):
        L = cp.asfortranarray(L)
    if L.flags.c_contiguous:
        lower = not lower  # Cusolver assumes F-order
        # For complex types, we need to conjugate the matrix
        if b.size < L.size:  # Conjugate the one with lower memory footprint
            b = b.conj()
        else:
            L = L.conj()
    if (b.dtype != dtype) or (not b.flags.f_contiguous):
        b = cp.asfortranarray(b)

    # Take correct dtype
    if dtype == 'f':
        potrs = _cusolver.spotrs
    elif dtype == 'd':
        potrs = _cusolver.dpotrs
    elif dtype == 'F':
        potrs = _cusolver.cpotrs
    elif dtype == 'D':
        potrs = _cusolver.zpotrs
    else:
        msg = ('dtype must be float32, float64, complex64 or complex128'
               ' (actual: {})'.format(L.dtype))
        raise ValueError(msg)

    handle = device.get_cusolver_handle()
    dev_info = cp.empty(1, dtype=cp.int32)

    potrs(
        handle,
        cublas.CUBLAS_FILL_MODE_LOWER if lower else
        cublas.CUBLAS_FILL_MODE_UPPER,
        L.shape[0],  # n, matrix size
        b.shape[1],  # nrhs
        L.data.ptr,
        L.shape[0],  # ldL
        b.data.ptr,
        b.shape[0],  # ldB
        dev_info.data.ptr)
    cp.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrs, dev_info)

    # Conjugate back if necessary
    if L.flags.c_contiguous and b.size < L.size:
        b = b.conj()
    return b.reshape(b_shape)


def cusolver_potrs_batched(L, b, lower: bool):
    """ Implements `cusolver_potrs` for batched input arrays """
    from cupy_backends.cuda.libs import cusolver as _cusolver
    import cupyx.cusolver

    if not cupyx.cusolver.check_availability('potrsBatched'):
        raise RuntimeError('potrsBatched is not available')

    # CHeck input arrays
    assert L.shape[-1] == L.shape[-2], "L is not a batch of square matrices"
    assert b.ndim >= L.ndim-1, "Batch dimension of b is different than that \
        of L"
    b_shape = b.shape
    if b.ndim < L.ndim:
        b = b[..., None]
    assert b.shape[:-2] == L.shape[:-2], \
        "Batch dimension of L and b do not match"

    # Check dtype and memory alignment
    dtype = np.promote_types(L.dtype, b.dtype)
    dtype = np.promote_types(dtype, 'f')
    L = L.astype(dtype, order='C', copy=False)
    b = b.astype(dtype, order='C', copy=False)
    assert L.flags.c_contiguous and b.flags.c_contiguous, \
        "Unexpected non C-contiguous arrays"
    lower = not lower  # Cusolver assumes F-order

    # Pick function handle
    if dtype == 'f':
        potrsBatched = _cusolver.spotrsBatched
    elif dtype == 'd':
        potrsBatched = _cusolver.dpotrsBatched
    elif dtype == 'F':
        potrsBatched = _cusolver.cpotrsBatched
    elif dtype == 'D':
        potrsBatched = _cusolver.zpotrsBatched
    else:
        msg = ('dtype must be float32, float64, complex64 or complex128'
               ' (actual: {})'.format(L.dtype))
        raise ValueError(msg)

    # Variables for potrs batched
    handle = device.get_cusolver_handle()
    dev_info = cp.empty(1, dtype=np.int32)
    batch_size = np.prod(L.shape[:-2])
    n = L.shape[-1]
    b = b.conj()
    L_p = cp._core._mat_ptrs(L)
    nrhs = b.shape[-1]

    # Allocate temporary working array in case nrhs > 1
    if nrhs == 1:
        b_tmp = b[..., 0]
    else:
        b_tmp = cp.empty(b.shape[:-1], dtype=b.dtype, order='C')
    b_tmp_p = cp._core._mat_ptrs(b_tmp[..., None])

    # potrs_batched supports only nrhs=1, so we have to loop over the nrhs
    for i in range(b.shape[-1]):

        if nrhs > 1:  # Copy results back to the original array
            b_tmp[...] = b[..., i]

        potrsBatched(
            handle,
            cublas.CUBLAS_FILL_MODE_LOWER if lower else
            cublas.CUBLAS_FILL_MODE_UPPER,
            n,  # n
            1,  # nrhs
            L_p.data.ptr,  # A
            L.shape[-2],  # lda
            b_tmp_p.data.ptr,  # Barray
            b_tmp.shape[-1],  # ldb
            dev_info.data.ptr,  # info
            batch_size  # batchSize
        )
        cp.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
            potrsBatched, dev_info)

        if nrhs > 1:  # Copy results back to the original array
            b[..., i] = b_tmp.conj()
        else:
            b = b_tmp.conj()

    # Return b in the original shape
    return b.reshape(b_shape)


def cdist_cuda(x, y):
    """ Calculates pairwise euclidean distance between input arrays. Supports batched arrays. 
    
    Parameters
    x : cp.ndarray of shape (*batch_shape, num_x, num_features)
        Input left array

    y : cp.ndarray of shape (*batch_shape, num_y, num_features)
        Input right array

    Returns

    cp.ndarray : array of shape (*batch_shape, num_x, num_y)
        Pairwise distance between points x and y. Array at index (i_batch, i_x, i_y) stores the distance
        between points x[i_batch, i_x, :] and y[i_batch, i_y, :]
    
    """
    # Check input
    assert x.ndim > 1, "X is not at least a 2d array"
    assert y.ndim > 1, "Y is not at least a 2d array"
    assert x.shape[:-2] == y.shape[:-2], "Number of batch dimensions mismatch"
    assert x.shape[-1] == y.shape[-1], "Number of features mismatch"
    assert max(np.prod(x.shape), np.prod(y.shape)) < np.iinfo(np.int64).max, "Integer overflow in cuda kernel"
    assert max(x.shape[-2], y.shape[-2], np.prod(x.shape[:-2])) < np.iinfo(np.int32).max, "Integer overflow in cuda kernel"

    batch_shape = x.shape[:-2]
    x = x.reshape((-1, *x.shape[-2:]))
    y = y.reshape((-1, *y.shape[-2:]))
    
    # Extract variables    
    kernel_defines = {}
    kernel_defines['NUM_FEATURES'] = x.shape[-1]
    num_batches = x.shape[0]
    kernel_defines['NUM_BATCHES'] = num_batches
    num_x = x.shape[-2]
    kernel_defines['NUM_X'] = num_x
    num_y = y.shape[-2]
    kernel_defines['NUM_Y'] = num_y
    dtype = cp.promote_types(x.dtype, y.dtype)
    dtype = cp.promote_types(dtype, 'f')
    if dtype == 'f':
        kernel_defines['DATA_TYPE'] = 'float'
    elif dtype == 'd':
        kernel_defines['DATA_TYPE'] = 'double'
    else:
        msg = ('dtype must be float32, float64, complex64 or complex128'
               ' (actual: {})'.format(dtype))
        raise ValueError(msg)

    # Swap axes for faster accessing
    x, y = cp.swapaxes(x, -1, -2), cp.swapaxes(y, -1, -2)
    
    # Esnure C contiguity
    x, y = cp.array(x, order='C', copy=False), cp.array(y, order='C', copy=False)
    assert x.flags.c_contiguous, "x is not c contiguous"
    assert y.flags.c_contiguous, "y is not c contiguous"

    # Allocate output
    out = cp.empty((x.shape[0], x.shape[-1], y.shape[-1]), dtype=dtype)

    # Kernel text
    kernel_text = """
    extern "C"
    __global__ void cdist(
        DATA_TYPE const * x,
        DATA_TYPE const * y,
        DATA_TYPE * dists
    )
    {
        int i_batch = blockIdx.x * blockDim.x + threadIdx.x;
        int i_x = blockIdx.y * blockDim.y + threadIdx.y;
        int i_y = blockIdx.z * blockDim.z + threadIdx.z;
        
        if ((i_batch >= NUM_BATCHES) || (i_x >= NUM_X) || (i_y >= NUM_Y))
            return;

        DATA_TYPE const * this_x = x + ((long long) i_batch) * NUM_FEATURES * NUM_X + i_x;
        DATA_TYPE const * this_y = y + ((long long) i_batch) * NUM_FEATURES * NUM_Y + i_y;
        DATA_TYPE dist = 0;
        DATA_TYPE tmp;

        for (int i=0; i<NUM_FEATURES; i++) {
            tmp = this_x[i * NUM_X] - this_y[i * NUM_Y];
            dist += tmp * tmp;    
        }
        dists[((long long) i_batch) * NUM_X * NUM_Y + ((long long) i_x) * NUM_Y + i_y] = dist;
    }
    """

    # Call kernel
    lines = ["#define " + k + " " + str(v) for k, v in kernel_defines.items()] + ["", ""]
    kernel_text = "\n".join(lines) + kernel_text
    krn = cp.RawKernel(kernel_text, 'cdist')
    block_size_y = min(64, max(1, int(2**np.ceil(np.log2(num_y)))))
    block_size_x = min(int(512/block_size_y), num_x)
    block_size = (1,block_size_x,block_size_y)
    grid_size = (
        num_batches,
        int(np.ceil(num_x / block_size_x)),
        int(np.ceil(num_y / block_size_y))
    )
    krn(grid_size, block_size, (x, y, out))
    cp.sqrt(out, out=out)

    out = out.reshape((*batch_shape, *out.shape[-2:]))
    return out
