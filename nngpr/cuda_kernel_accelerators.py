""" This module implements specific operations on batched kernels that are optmized for Cuda hardware through
custom Cuda code. Specifically, this module provides optimized functions for nearest neighbour search and kernel 
evaluations.
"""

import numbers
import copy

import numpy as np
import cupy as cp
from sklearn.gaussian_process import kernels


class CudaKernelAccelerator:
    """ Class that provides the end-user interface to NngprWorker instances. This class collects the different 
    kernel optimization classes (currently only RBF is available) and dispatches the accelerated functions to the right
    kernel class
    """

    def __init__(self):
        super().__init__()
        self._last_kernel = None
        self._last_theta = np.nan
        self._last_params = None
        self._last_mapping = None
        self._acc_fn = None

    def can_accelerate_kernel(self, kernel):
        """ Indicates whether accelerated functions are available for the given kernel 
        
        Parameters
        ----------
        kernel : nngpr.base_nngpr.NngprWorker
            Gaussian process kernel

        Returns
        -------

        bool : True if accelerated functions are available for the kernel in input
        
        """
        self._update_kernel_if_needed(kernel)
        return self._acc_fn is not None

    def get_kernel_params(self, kernel):
        """ Helper functions that extract the kernel parameters and places them in a dictionary. 

        Parameters
        ----------
        kernel : nngpr.base_nngpr.NngprWorker
            Input kernel whose parameters will be extracted and returned as output
        
        Returns
        -------
        kernel_parameters : dict
            Dictionary that maps each kernel parameter (name) to its value, e.g.:
            {'noise_level': 1.234, 'length_scale': [1.8,3.3,6.2,8.3]}

        """
        self._update_kernel_if_needed(kernel)
        return self._last_params

    def get_kernel_theta_mapping(self, kernel):
        """ Helper functions that maps each parameter retruned by 'get_kernel_params' to the corresponding value 
        in kernel.theta

        Parameters
        ----------
        kernel : nngpr.base_nngpr.NngprWorker
            Input kernel whose parameters will be mapped to kernel.theta
        
        Returns
        -------
        kernel_parameters : dict
            Dictionary that maps each kernel parameter (name) to its index in kernel.theta, e.g.
            {'noise_level': 0, 'length_scale': [1,2,3,4]}

        """
        self._update_kernel_if_needed(kernel)
        return self._last_mapping
    
    def find_nearest_neighbors(self, cuda_device_index, ref, query, nn_type, kernel, num_nn, ref_cut_index):
        """ Implements nngpr.base_nngpr.WorkerNngpr.find_nearest_neighbors

        Parameters
        ----------

        cuda_device_index : int
            Cuda device to use for calculations
        
        ref : any array type implementing the dlpack interface.
            see nngpr.base_nngpr.WorkerNngpr docstring.

        query : any array type implementing the dlpack interface.
            see nngpr.base_nngpr.WorkerNngpr docstring

        nn_type : see nngpr.base_nngpr.WorkerNngpr docstring

        kernel : see nngpr.base_nngpr.WorkerNngpr docstring

        num_nn : see nngpr.base_nngpr.WorkerNngpr docstring

        ref_cut_index : see nngpr.base_nngpr.WorkerNngpr docstring

        Returns
        -------
        Same as nngpr.base_nngpr.WorkerNngpr.find_nearest_neighbors. Raises NotImplementedError if the kernel cannot be accelerated.
        
        """
        if not self.can_accelerate_kernel(kernel):
            raise NotImplementedError("Cannot accelerate this kernel")
        return self._acc_fn[0](cuda_device_index, ref, query, nn_type, kernel, num_nn, ref_cut_index)
    
    def fill_nn_kernel(self, cuda_device_index, x, nn_indices, kernel, eval_gradient=False):
        """ Implements nngpr.base_nngpr.WorkerNngpr.fill_nn_kernel

        Parameters
        ----------

        cuda_device_index : int
            Cuda device to use for calculations
        
        x : any array type implementing the dlpack interface.
            see nngpr.base_nngpr.WorkerNngpr docstring

        nn_indices : any array type implementing the dlpack interface.
            see nngpr.base_nngpr.WorkerNngpr docstring

        kernel : see nngpr.base_nngpr.WorkerNngpr docstring

        eval_gradient : see nngpr.base_nngpr.WorkerNngpr docstring

        Returns
        -------
        Same as nngpr.base_nngpr.WorkerNngpr.fill_nn_kernel. Raises NotImplementedError if the kernel cannot be accelerated.
        
        """
        if not self.can_accelerate_kernel(kernel):
            raise NotImplementedError("Cannot accelerate this kernel")
        return self._acc_fn[1](cuda_device_index, x, nn_indices, kernel, eval_gradient=eval_gradient)
    
    def _update_kernel_if_needed(self, kernel):
        """ Extracts the kernel parameters and theta mapping for this kernel and stores it in instance attributes """
        if self._last_kernel is kernel and np.all(kernel.theta == self._last_theta):
            return
        self._acc_fn = None
        self._last_kernel = kernel
        self._last_theta = copy.deepcopy(self._last_kernel.theta)
        self._last_params = None
        self._last_mapping = None

        try:
            params = RbfAccelerator.get_kernel_params(kernel)
            theta_mapping = RbfAccelerator.get_kernel_theta_mapping(kernel)
            self._last_params = params
            self._last_mapping = theta_mapping
            self._acc_fn = (RbfAccelerator.find_nearest_neighbors, RbfAccelerator.fill_nn_kernel)

        except NotImplementedError:
            pass


class RbfAccelerator:
    """ This class implements Cuda accelerated functions for kernles that can be casted to the form:
        WhiteKernel + ConstantKernel * RBF
    """

    @staticmethod
    def can_accelerate_kernel(kernel):
        """ Indicates whether accelerated functions are available for the given kernel. Only kernels
        of the form WhiteKernel + ConstantKernel * RBF can be accelerated by this class
        
        Parameters
        ----------
        kernel : nngpr.base_nngpr.NngprWorker
            Gaussian process kernel

        Returns
        -------

        bool : True if accelerated functions are available for the kernel in input
        
        """
        try:
            RbfAccelerator.get_kernel_params(kernel)
            RbfAccelerator.get_kernel_theta_mapping(kernel)
        except NotImplementedError:
            return False
        
        return True
        
    @staticmethod
    def find_nearest_neighbors(cuda_device_index, ref, query, nn_type, kernel, num_nn, ref_cut_index):
        """ Implements nngpr.base_nngpr.WorkerNngpr.find_nearest_neighbors

        Parameters
        ----------

        cuda_device_index : int
            Cuda device to use for calculations
        
        ref : any array type implementing the dlpack interface.
            see nngpr.base_nngpr.WorkerNngpr docstring.

        query : any array type implementing the dlpack interface.
            see nngpr.base_nngpr.WorkerNngpr docstring

        nn_type : see nngpr.base_nngpr.WorkerNngpr docstring

        kernel : see nngpr.base_nngpr.WorkerNngpr docstring

        num_nn : see nngpr.base_nngpr.WorkerNngpr docstring

        ref_cut_index : see nngpr.base_nngpr.WorkerNngpr docstring

        Returns
        -------
        Same as nngpr.base_nngpr.WorkerNngpr.find_nearest_neighbors. Raises NotImplementedError if the kernel cannot be accelerated.
        
        """

        kernel_params = RbfAccelerator.get_kernel_params(kernel)

        with cp.cuda.Device(cuda_device_index):

            # Check if input is cupy array
            ref, query = _ensure_cupy_array(ref), _ensure_cupy_array(query)
            
            data_type = np.promote_types(ref.dtype, query.dtype)

            if nn_type == 'kernel-space' and 'length_scale' in kernel_params:
                ls = kernel_params['length_scale']
                if type(ls) is np.ndarray:
                    ls = cp.array(ls, dtype=data_type)
                ref, query = ref / ls, query / ls

            # Start building the kernel header
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
            if ref_cut_index is None:
                defines['MAX_SEARCH_POINT'] = defines['N_REF']
            else:
                defines['MAX_SEARCH_POINT'] = '(i_query + ' + str(ref_cut_index) + ')'
                index_gpu[:, :max(0, num_nn - ref_cut_index)] = -1

            # Build the kernel with the desired options
            defines['BLOCK_DIM'] = block_dim_nn
            defines['QUERY_PITCH'] = query_pitch
            defines['NUM_K_FOR_CACHE'] = k_for_cache
            defines['BLOCK_DIM'] = block_dim_nn
            defines['INDEX_PITCH'] = index_gpu.shape[1]

            kernel_headers = kernel_headers + ["#define " + k + " " + str(v) for k, v in defines.items()] + ["", ""]
            kernel_headers = "\n".join(kernel_headers)
            krn = cp.RawKernel(kernel_headers + _calc_nn_rbf_kernel_text, 'calc_nn')

            # Run
            if dynamic_shared_mem_size > 0:
                krn.max_dynamic_shared_size_bytes = dynamic_shared_mem_size
            krn(
                grid_size, block_size, 
                (ref, query, index_gpu, *opt_call_k_args), **opt_call_kwargs)
            return cp.transpose(index_gpu)

    @staticmethod
    def fill_nn_kernel(cuda_device_index, x, nn_indices, kernel, eval_gradient=False):
        """ Implements nngpr.base_nngpr.WorkerNngpr.fill_nn_kernel

        Parameters
        ----------

        cuda_device_index : int
            Cuda device to use for calculations
        
        x : any array type implementing the dlpack interface.
            see nngpr.base_nngpr.WorkerNngpr docstring

        nn_indices : any array type implementing the dlpack interface.
            see nngpr.base_nngpr.WorkerNngpr docstring

        kernel : see nngpr.base_nngpr.WorkerNngpr docstring

        eval_gradient : see nngpr.base_nngpr.WorkerNngpr docstring

        Returns
        -------
        Same as nngpr.base_nngpr.WorkerNngpr.fill_nn_kernel. Raises NotImplementedError if the kernel cannot be accelerated.
        
        """

        with cp.cuda.Device(cuda_device_index):

            kernel_params = RbfAccelerator.get_kernel_params(kernel)
            kernel_theta_mapping = RbfAccelerator.get_kernel_theta_mapping(kernel)
            nn_indices = nn_indices.T  # For faster cuda access

            # Check if input is cupy array
            x, nn_indices = _ensure_cupy_array(x), _ensure_cupy_array(nn_indices)
            
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
            kernel_text = "\n".join(lines) + _fill_nn_RBF_kernel_text

            # Execute kernel
            krn = cp.RawKernel(kernel_text, 'fill_nn_kernel')
            block_size = (block_dim_nn,)
            grid_size = (int(np.ceil(nn_indices.shape[1] / block_size[0])),)

            krn(grid_size, block_size, (
                x, nn_indices, rbf_length_scales, nn_kernel, nn_kernel_grad,
                x.dtype.type(kernel_params.get('constant_value', 0)),
                x.dtype.type(kernel_params.get('noise_level', 0)),
                x.dtype.type(kernel_params.get('rbf_level', 1 if 'length_scale' in kernel_params else 0))))

            # Post-processing
            nn_kernel = cp.moveaxis(nn_kernel, -1, 0)  # Follow sickitlearn convention
            nn_kernel_grad = cp.moveaxis(nn_kernel_grad, -1, 0)  # Follow sickitlearn convention

            return nn_kernel, nn_kernel_grad

    @staticmethod
    def get_kernel_params(kernel):
        """ Helper functions that extract the kernel parameters and places them in a dictionary. 

        Parameters
        ----------
        kernel : nngpr.base_nngpr.NngprWorker
            Input kernel whose parameters will be extracted and returned as output
        
        Returns
        -------
        kernel_parameters : dict
            Dictionary that maps each kernel parameter (name) to its value, e.g.:
            {'noise_level': 1.234, 'length_scale': [1.8,3.3,6.2,8.3]}

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
            p1 = RbfAccelerator.get_kernel_params(kernel.k1)
            p2 = RbfAccelerator.get_kernel_params(kernel.k2)
            # Sum is supported only with a white kernel
            is_p1_rbf = len(set(p1.keys()).difference({'noise_level', 'constant_value'})) > 0
            is_p2_rbf = len(set(p2.keys()).difference({'noise_level', 'constant_value'})) > 0
            if is_p1_rbf and is_p2_rbf:
                raise NotImplementedError("Sum of kernels is not supported between RBF kernels. Got sum between "
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
            p1 = RbfAccelerator.get_kernel_params(kernel.k1)
            p2 = RbfAccelerator.get_kernel_params(kernel.k2)
            # Product is supported only with a ConstantKernel
            is_p1_const = len(set(p1.keys()).difference({'constant_value'})) == 0
            is_p2_const = len(set(p2.keys()).difference({'constant_value'})) == 0
            if not is_p1_const and not is_p2_const:
                raise NotImplementedError("Product of kernels is supported only with a ConstantKernel. Got product between "
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

        raise NotImplementedError("Unsupported kernel type: " + str(type(kernel)))
    
    @staticmethod
    def get_kernel_theta_mapping(kernel):
        """ Helper functions that maps each parameter retruned by 'get_kernel_params' to the corresponding value 
        in kernel.theta

        Parameters
        ----------
        kernel : nngpr.base_nngpr.NngprWorker
            Input kernel whose parameters will be mapped to kernel.theta
        
        Returns
        -------
        kernel_parameters : dict
            Dictionary that maps each kernel parameter (name) to its index in kernel.theta, e.g.
            {'noise_level': 0, 'length_scale': [1,2,3,4]}

        """

        n_theta = len(kernel.theta)
        epsi = 1e-15
        kernel = kernel.clone_with_theta(np.arange(n_theta))
        expected = np.exp(kernel.theta)
        kernel_params = RbfAccelerator.get_kernel_params(kernel)

        def find_index_for_param(param_name):
            param = kernel_params[param_name]
            if (param_name != 'length_scale') or (isinstance(param, numbers.Number)):
                found = (param < expected * (1 + epsi)) & (param > expected * (1 - epsi))
                n_found = np.sum(found)
                if n_found > 1:
                    raise NotImplementedError("Cannot find theta mapping for parameter " + param_name)
                if n_found == 0:
                    return None
                return np.where(found)[0][0]

            params = kernel_params[param_name].reshape(-1)
            found = np.zeros(len(expected), dtype=bool)
            for param in params:
                tmp = (param < expected * (1 + epsi)) & (param > expected * (1 - epsi))
                if np.sum(tmp) == 1:
                    found[np.where(tmp)[0][0]] = 1
            n_found = np.sum(found)
            if n_found > params.size:
                raise NotImplementedError("Cannot find theta mapping for parameter " + param_name)
            if n_found == 0:
                return None
            ind = np.where(found)[0]
            assert (len(ind) == 1) or (set(np.unique(np.diff(ind))) == {1}), "Unsupported case of non-adjacent parameters for length scale"
            return ind

        theta_mapping = {param: find_index_for_param(param) for param in kernel_params}
        theta_mapping = {k: v for k, v in theta_mapping.items() if v is not None}
        num_params_found = sum(len(v) if type(v) is np.ndarray else 1 for k, v in theta_mapping.items())
        if num_params_found < len(kernel.theta):
            raise NotImplementedError("Cannot find the mapping for all kernel thetas")
        assert num_params_found == len(kernel.theta), "Unexpected case when mapping kernel thetas to parameters"

        return theta_mapping


def _ensure_cupy_array(x):
    """ Utility function that converts an input array to a cupy array through
    the 'from_dlpack' interface """
    if type(x) is cp.ndarray:
        return x
    return cp.from_dlpack(x)


# TODO: implent other kernels, e.g. Matern

# Cuda kernel text to calculate nearest neighbors with RBF kernels
_calc_nn_rbf_kernel_text = """
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

# Cuda kernel text to calculate the GP kernel and its gradient for RBF kernels
_fill_nn_RBF_kernel_text = """
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
