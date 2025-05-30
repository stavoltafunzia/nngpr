""" Implements Nngpr using Numpy arrays """

import os
from numbers import Integral

import numpy as np
from scipy import linalg as sl
from scipy.spatial.distance import cdist
from sklearn.utils._param_validation import Interval
from sklearn.utils import check_random_state

from .base_nngpr import Nngpr, NngprWorker, ArrayModule


class NumpyNngpr(Nngpr):
    """ This class is the Nngpr end-user interface that uses a Numpy-based backend.
    
    Parameters
    ----------

    kernel : see parent class docstring

    num_workers: int | None
        Number of numpy workers to use for Nngpr calculation. If None, the number of workers is set equal to the
        number of CPU cores.

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
        "num_workers": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
            self,
            kernel=None,
            *,
            num_workers=None,
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
            random_gen_batch_byte_size=None):

        self.num_workers = os.cpu_count() if num_workers is None else num_workers
        workers = [NumpyWorker()] * self.num_workers
        super().__init__(
            workers, num_nn=num_nn, kernel=kernel, alpha=alpha, optimizer=optimizer, 
            n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y, 
            copy_X_train=copy_X_train, random_state=random_state, batch_size=batch_size,
            nn_type=nn_type, allow_downcast_f32=allow_downcast_f32, distribute_method=distribute_method,
            mp_start_method=mp_start_method, random_gen_batch_byte_size=random_gen_batch_byte_size
            )


class NumpyProxy(ArrayModule):
    """ Implements a Numpy-baed array module. Thin wrapper around numpy which only adds support
    for the batched cdist function 
    """
    def __getattr__(self, name):
        if name == 'cdist':
            return self.cdist
        return getattr(np, name)

    def to_numpy_impl(self, x):
        """ Implements parent class abstract method. See parent class docstring """
        return x

    def from_numpy(self, x):
        """ Implements parent class abstract method. See parent class docstring """
        return x
    
    def to_float32(self, x):
        """ Implements parent class abstract method. See parent class docstring """
        return x.astype(np.float32, copy=False)
    
    def cdist(self, x, y):
        """ Implements parent class abstract method. See parent class docstring """
        
        shape = x.shape[:-2]
        out = np.empty((*shape, x.shape[-2], y.shape[-2]), dtype=x.dtype)
        for index in np.ndindex(shape):
            out[*index] = cdist(x[*index], y[*index])
        return out


class NumpyWorker(NngprWorker):

    """ Implements NngprWorker with Numpy arrays """

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.np = NumpyProxy()

    def get_array_module(self):
        """ Implements parent class abstract method. See parent class docstring """
        return self.np

    def clear_cache(self):
        """ Implements parent class abstract method. See parent class docstring """
        pass

    def batched_chosolve(self, L, y, lower, may_overwerite_b=False):
        """ Implements parent class abstract method. See parent class docstring """

        assert len(L.shape) > 2, "Not a batch of Cholesky decompositions"
        assert L.shape[-2] == L.shape[-1], "Not square matrices"
        assert len(y.shape) == len(L.shape), "Batch or rhs size mismatch"

        shape = np.broadcast_shapes(L.shape[:-2], y.shape[:-2])
        out = np.empty_like(y)

        L_loop_mask = tuple(x == 1 for x in L.shape[:-2])
        y_loop_mask = tuple(x == 1 for x in y.shape[:-2])
        for index in np.ndindex(shape):
            y_index = (0 if mask else i  for mask, i in zip(y_loop_mask, index))
            L_index = (0 if mask else i for mask, i in zip(L_loop_mask, index))
            out[index] = sl.cho_solve((L[*L_index], lower), y[*y_index], overwrite_b=may_overwerite_b)

        return out

    def batched_chofactor(self, x, may_overwrite_x=False):
        """ Implements parent class abstract method. See parent class docstring """
        
        assert len(x.shape) > 2, "Not enough dimensions"
        assert x.shape[-2] == x.shape[-1], "Not square matrices"

        shape = x.shape[:-2]
        out = x if may_overwrite_x else np.empty_like(x)

        for index in np.ndindex(shape):
            out[*index] = sl.cho_factor(x[*index], lower=True, overwrite_a=may_overwrite_x)[0]
        return out, True
    
    def check_random_state(self, random_state):
        """Turn random_state into a np.random.Generator instance.

        Parameters
        ----------
        random_state : None, int or instance of np.random.Generator
            If random_state is None, returns np.random.default_rng()
            If random_state is an int, returns np.random.default_rng(random_state)
            If random_state is already an instance np.random.Generator, the same instance is returned
            Otherwise raise ValueError.

        Returns
        -------
        :class:`np.random.Generator
            The random generator object based on `random_state` parameter.

        """
        return check_random_state(random_state)

    def random_normal(self, random_state, mu, sigma, shape):
        """ Implements parent class abstract method. See parent class docstring """
        return random_state.normal(loc=mu, scale=sigma, size=shape)
