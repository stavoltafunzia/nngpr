import functools
from numbers import Integral
from collections.abc import Hashable
import importlib
import warnings

import numpy as np

try:
    import torch
except ImportError:
    raise ImportError("This module requires torch. Install with `pip install nngpr[torch]`")

from .base_nngpr import Nngpr, NngprWorker, ArrayModule


class TorchNngpr(Nngpr):

    """ This class is the Nngpr end-user interface that uses a Torch-based backend.
    
    Parameters
    ----------

    torch_devices : list[str]
        List of torch devices to be used, e.g. ['cuda:0', 'cuda:1', 'cpu']. Etherogeneous devices can be used together,
        e.g. GPUs can be used next to CPUs. Note that using etherogeneous devices may cause a slight overhead due to
        the switch to the 'multiprocessing-heterogenous' distribute method (see parent class docstring for 
        'distribute_method') unless a different distribute_method is specified by the user. However, for configurations with
        few GPUs and powerful CPUs, using both GPUs and CPUs can lead to performance increase.

    kernel : see parent class docstring

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
        **Nngpr._parameter_constraints
    }

    def __init__(
            self,
            torch_devices,
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

        kernel = self.get_default_kernel() if kernel is None else kernel
        if type(torch_devices) is str:
            if torch_devices == 'cuda':
                torch_devices = ['cuda:' + str(i) for i in range(torch.cuda.device_count())]
            else:
                torch_devices = [torch_devices]
        self.torch_devices = torch_devices
        workers = [TorchWorker(dev, kernel) for dev in torch_devices]
        is_heterogenous = len(set(x.device_type for x in workers)) > 1
        if is_heterogenous and (distribute_method is None):
            distribute_method = 'multiprocessing-heterogenous'
        super().__init__(
            workers, num_nn=num_nn, kernel=kernel, alpha=alpha, optimizer=optimizer, 
            n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y, copy_X_train=copy_X_train, 
            random_state=random_state, batch_size=batch_size,nn_type=nn_type, allow_downcast_f32=allow_downcast_f32, 
            distribute_method=distribute_method, mp_start_method=mp_start_method, 
            random_gen_batch_byte_size=random_gen_batch_byte_size, distribute_environ=distribute_environ
            )


class TorchProxy(ArrayModule):

    """ Implements a Torch-based array module. Thin wrapper around torch which takes care of transalting numpy dtypes to torch 
    dtypes and allocates arrays on the selected device. It also converts torch.sort to the same signature as numpy.sort.

    torch_device
    ----------

    torch_device : str
        Torch device to use by this class instance

    """

    dtype_map = {
        np.uint8: torch.uint8,
        np.uint16: torch.uint16,
        np.uint32: torch.uint32,
        np.uint64: torch.uint64,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,

        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.uint16): torch.uint16,
        np.dtype(np.uint32): torch.uint32,
        np.dtype(np.uint64): torch.uint64,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.int16): torch.int16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.complex64): torch.complex64,
        np.dtype(np.complex128): torch.complex128,
    }

    def __init__(self, torch_device):
        self.torch_device = torch_device
        self.overrides_map = {
            'sort': self.sort,
        }

    def to_numpy_impl(self, x):
        """ Implements parent class abstract method. See parent class docstring """
        return x.cpu().numpy()

    def from_numpy(self, x):
        """ Implements parent class abstract method. See parent class docstring """
        return torch.asarray(x, device=self.torch_device)
    
    def to_float32(self, x):
        """ Implements parent class abstract method. See parent class docstring """
        if x.dtype == torch.float32:
            return x
        return x.to(dtype=torch.float32)

    @staticmethod
    def convert_args(*args, **kwargs):
        """ Converts numpy dtypes to torch dtypes in the args or kwargs """
        args = [TorchProxy.dtype_map.get(x, x) if isinstance(x, Hashable) else x for x in args]
        if 'dtype' in kwargs:
            kwargs['dtype'] = TorchProxy.dtype_map.get(kwargs['dtype'], kwargs['dtype'])
        return args, kwargs
    
    def redirect_with_device(self, name, *args, **kwargs):
        """ Executes the torch callable named `name` with arguments *args and **kwargs inside the device 'torch_device'"""
        assert 'device' not in kwargs
        args, kwargs = self.convert_args(*args, **kwargs)
        return getattr(torch, name)(*args, device=self.torch_device, **kwargs)
    
    def generic_redirect(self, attr, *args, **kwargs):
        """ Executes the torch callable 'attr' converting numpy dtypes to torch dtypes in the args or kwargs'"""
        args, kwargs = self.convert_args(*args, **kwargs)
        return attr(*args, **kwargs)
    
    def sort(self, *args, **kwargs):
        """ Wrapper around torch.sort to return only the sorted array, a-la numpy """
        args, kwargs = self.convert_args(*args, **kwargs)
        return torch.sort(*args, **kwargs)[0]
    
    def cdist(self, x, y):
        """ Implements parent class abstract method. See parent class docstring """
        return torch.cdist(x, y)
    
    def __getattr__(self, name):
        """ Exposes torch attributes in this class """
        if (fn := self.overrides_map.get(name)) is not None:
            return fn
        
        if name in {'empty', 'zeros', 'ones', 'full', 'asarray', 'eye', 'arange'}:
            return functools.partial(self.redirect_with_device, name)
        
        attr = getattr(torch, name)
        if not callable(attr):
            return attr
        
        return functools.partial(self.generic_redirect, attr)
    
    def __setstate__(self, state):
        self.__dict__.update(state)


class TorchWorker(NngprWorker):

    """ Implements NngprWorker with Cupy arrays 
    
    Parameters
    ----------

    torch_device : str
        Torch device to use for the worker

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

    xp : TorchProxy
        TorchProxy instance

    accelerate_kernel : bool | None
        Same as accelerate_kernel arguments

    kernel_acc : CudaKernelAccelerator
        CudaKernelAccelerator instance
    
    """

    def __init__(self, torch_device, kernel, random_gen_batch_byte_size=None, nn_large_size_preferred=None, 
                 accelerate_kernel=None, **super_kwargs):
        
        self.device_type = torch.device(torch_device).type
        self.total_memory_bytes = None
        self.xp = TorchProxy(torch_device)
        self.accelerate_kernel = accelerate_kernel
        self.kernel_acc = None
        self.cuda_device_index = None
        if self.device_type == 'cuda':
            try:
                cuda_kern_acc = importlib.import_module('nngpr.cuda_kernel_accelerators')
                self.kernel_acc = cuda_kern_acc.CudaKernelAccelerator()
            except ImportError:
                warnings.warn("""When using cuda devices, it's highly recommended to install cupy (with cusolver and cublas) 
                              to exploit cuda-optimized kernels""")
                assert self.accelerate_kernel is not True, "Kernel acceleration requries cupy (with cublas and cusolver)"
                self.accelerate_kernel = False
            total_memory_bytes = torch.cuda.get_device_properties(torch_device).total_memory
            if random_gen_batch_byte_size is None:
                random_gen_batch_byte_size = total_memory_bytes // 4
            self.total_memory_bytes = total_memory_bytes
        else:
            assert self.accelerate_kernel is not True, "Kernel acceleration is implemented only on Cuda devices"
            self.accelerate_kernel = False
        if (nn_large_size_preferred is None) and (self.accelerate_kernel is not False) and (self.kernel_acc.can_accelerate_kernel(kernel)):
            nn_large_size_preferred = 25000  # Accelerated nearest neighbor kernel works faster with large batches. This number is
                                             # based on heuristic
        super().__init__(random_gen_batch_byte_size=random_gen_batch_byte_size, 
                         nn_large_size_preferred=nn_large_size_preferred, **super_kwargs)
        
    def get_array_module(self):
        """ Implements parent class abstract method. See parent class docstring """
        return self.xp

    def clear_cache(self):
        """ Implements parent class abstract method. See parent class docstring """
        if self.device_type == 'cuda':
            torch.cuda.empty_cache()

    def batched_chosolve(self, L, y, lower, may_overwerite_b=False):
        """ Implements parent class abstract method. See parent class docstring """
        return torch.cholesky_solve(y, L, upper=not lower)

    def batched_chofactor(self, x, may_overwrite_x=False):
        """ Implements parent class abstract method. See parent class docstring """
        return torch.linalg.cholesky(x, upper=False), True

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
            self.acc_find_nearest_neighbors, ref, query, nn_type, kernel, num_nn, ref_cut_index)

        # Note: the accelerate function is slower than the base one when the number of query points is less that roughly 20000. However,
        # the base function has some memory-usage issues (due to Cupy) caused apparently by not releasing unused memory and may result
        # in out of memory errors. For this reason, the accelerated function is preferred.
        
        return self._accelerate_if_needed(kernel, base_fn, acc_fn)
    
    def fill_nn_kernel(self, x, nn_indices, kernel, eval_gradient=False):
        """ Overrides parent class method. See parent class docstring """
        base_fn = functools.partial(
            super().fill_nn_kernel, x, nn_indices, kernel, eval_gradient=eval_gradient)
        
        acc_fn = functools.partial(
            self.acc_fill_nn_kernel, x, nn_indices, kernel, eval_gradient=eval_gradient)
        
        return self._accelerate_if_needed(kernel, base_fn, acc_fn)

    def acc_find_nearest_neighbors(self, ref, query, nn_type, kernel, num_nn, ref_cut_index):
        res = self.kernel_acc.find_nearest_neighbors(self.cuda_device_index, ref, query, nn_type, kernel, num_nn, ref_cut_index)
        return torch.from_dlpack(res)

    def acc_fill_nn_kernel(self, x, nn_indices, kernel, eval_gradient):
        k, k_grad = self.kernel_acc.fill_nn_kernel(self.cuda_device_index, x, nn_indices, kernel, eval_gradient=eval_gradient)
        return torch.from_dlpack(k), torch.from_dlpack(k_grad)
    
    def check_random_state(self, seed):
        """Turn seed into a torch.Generator

        Parameters
        ----------
        seed : None, int or instance of torch.Generator
            If seed is None, returns torch.Generator(device==self.xp.torch_device)
            If seed is an int, sets the manual seed of the generator to seed
            If seed is already an instance np.random.Generator, the same instance is returned
            Otherwise raise ValueError.

        Returns
        -------
        :class:`np.random.Generator
            The random generator object based on `random_state` parameter.

        """
        if (seed is None) or isinstance(seed, Integral):
            rng = torch.Generator(device=self.xp.torch_device)
            if seed is not None:
                rng.manual_seed(seed)
            return rng
        if not isinstance(seed, torch.Generator):
            raise ValueError("Seed is not a torch.Generator instance or an integer")
        
        return seed

    def random_normal(self, random_state, mu, sigma, shape):
        """ Implements parent class abstract method. See parent class docstring """
        mu, sigma = self.xp.from_numpy(mu).expand(shape), self.xp.from_numpy(sigma).expand(shape)
        return torch.normal(mu, sigma, generator=random_state)
    
    def custom_batch_size_for_nn(self, num_nn, itemsize, kernel):
        """ Implements parent class abstract method. See parent class docstring """

        if self.device_type != 'cuda':
            return super().custom_batch_size_for_nn(num_nn, itemsize, kernel)
        
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
