"""Kernels that extends sklearn.gaussian_processes.kernels by allowing for custom array modules to be used (e.g.
Numpy, Cupy, Torch, etc) and supporting for batched operations """

import numpy as np
from sklearn.gaussian_process import kernels
from sklearn.base import clone
import math


class BatchedKernel(kernels.Kernel):
    """ Base class for Batched kernels """
    def __add__(self, b):
        if not isinstance(b, BatchedKernel):
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b):
        if not isinstance(b, BatchedKernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        if not isinstance(b, BatchedKernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b):
        if not isinstance(b, BatchedKernel):
            return Product(ConstantKernel(b), self)
        return Product(b, self)
    
    def __pow__(self, b):
        return Exponentiation(self, b)
    
    def bind_to_array_module(self, xp):
        res = BatchedKernelWrapper(self)
        res.xp = xp
        return res


class BatchedKernelWrapper(kernels.Kernel):
    """ Wrapper around a BatchedKernel that binds the kernel to the provided array module and allows to use the kernel 
    with a scikit-learn like interface 

    Parameters
    ----------

    base_kernel : BatchedKernel
        Kernel to wrap
    """

    def __init__(self, base_kernel):
        self.base_kernel = base_kernel
        self.xp = None

    def __call__(self, *args, **kwargs):
        return self.base_kernel(self.xp, *args, **kwargs)

    def diag(self, *args, **kwargs):
        return self.base_kernel.diag(self.xp, *args, **kwargs)
    
    def is_stationary(self):
        return self.base_kernel.is_stationary()
    
    @property
    def theta(self):
        return self.base_kernel.theta
    
    @theta.setter
    def theta(self, theta):
        self.base_kernel.theta = theta

    @property
    def bounds(self):
        return self.base_kernel.bounds

    def clone_with_theta(self, theta):
        res = BatchedKernelWrapper(self.base_kernel.clone_with_theta(theta))
        res.xp = self.xp
        return res

    def __getattr__(self, name):
        return getattr(self.base_kernel, name)
    
    def __repr__(self):
        return self.base_kernel.__repr__()
    
    def __str__(self):
        return self.base_kernel.__str__()
    
    def get_params(self, deep=True):
        return self.base_kernel.get_params(deep=deep)

    def set_params(self, **params):
        self.base_kernel.set_params(**params)
        return self

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter specifications."""
        return self.base_kernel.hyperparameters
    
    def __sklearn_clone__(self):
        base_clone = clone(self.base_kernel)
        res = BatchedKernelWrapper(base_clone)
        res.xp = self.xp
        return res


class Sum(BatchedKernel, kernels.Sum):
    """The `Sum` kernel takes two kernels :math:`k_1` and :math:`k_2`
    and combines them via

    .. math::
        k_{sum}(X, Y) = k_1(X, Y) + k_2(X, Y)

    Note that the `__add__` magic method is overridden, so
    `Sum(RBF(), RBF())` is equivalent to using the + operator
    with `RBF() + RBF()`.


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    k1 : Kernel
        The first base-kernel of the sum-kernel

    k2 : Kernel
        The second base-kernel of the sum-kernel

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import RBF, Sum, ConstantKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = Sum(ConstantKernel(2), RBF())
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    1.0
    >>> kernel
    1.41**2 + RBF(length_scale=1)
    """

    def __call__(self, xp, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (*batch_shape, n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (*batch_shape, n_samples_X, n_features) or list of object,\
                default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (*batch_shape, n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (*batch_shape, n_samples_X, n_samples_X, n_dims),\
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if eval_gradient:
            K1, K1_gradient = self.k1(xp, X, Y, eval_gradient=True)
            K2, K2_gradient = self.k2(xp, X, Y, eval_gradient=True)
            return K1 + K2, xp.concatenate((K1_gradient, K2_gradient), axis=-1)
        else:
            return self.k1(xp, X, Y) + self.k2(xp, X, Y)
        
    def diag(self, xp, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to `np.diag(self(X))`; however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (*batch_shape, n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (*batch_shape, n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.k1.diag(xp, X) + self.k2.diag(xp, X)


class Product(BatchedKernel, kernels.Product):
    """The `Product` kernel takes two kernels :math:`k_1` and :math:`k_2`
    and combines them via

    .. math::
        k_{prod}(X, Y) = k_1(X, Y) * k_2(X, Y)

    Note that the `__mul__` magic method is overridden, so
    `Product(RBF(), RBF())` is equivalent to using the * operator
    with `RBF() * RBF()`.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    ParametersX
    ----------
    k1 : Kernel
        The first base-kernel of the product-kernel

    k2 : Kernel
        The second base-kernel of the product-kernel


    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import (RBF, Product,
    ...            ConstantKernel)
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = Product(ConstantKernel(2), RBF())
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    1.0
    >>> kernel
    1.41**2 * RBF(length_scale=1)
    """

    def __call__(self, xp, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (*batch_shape, n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (*batch_shape, n_samples_Y, n_features) or list of object,\
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (*batch_shape, n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (*batch_shape, n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if eval_gradient:
            K1, K1_gradient = self.k1(xp, X, Y, eval_gradient=True)
            K2, K2_gradient = self.k2(xp, X, Y, eval_gradient=True)
            return K1 * K2, xp.concatenate(
                (K1_gradient * K2[..., None], K2_gradient * K1[..., None]), axis=-1)
        else:
            return self.k1(xp, X, Y) * self.k2(xp, X, Y)
        
    def diag(self, xp, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (*batch_shape, n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (*batch_shape, n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.k1.diag(xp, X) * self.k2.diag(xp, X)


class CompoundKernel(kernels.Kernel):
    """Kernel which is composed of a set of other kernels.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernels : list of Kernels
        The other kernels

    Examples
    --------
    >>> from sklearn.gaussian_process.kernels import WhiteKernel
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> from sklearn.gaussian_process.kernels import CompoundKernel
    >>> kernel = CompoundKernel(
    ...     [WhiteKernel(noise_level=3.0), RBF(length_scale=2.0)])
    >>> print(kernel.bounds)
    [[-11.51292546  11.51292546]
     [-11.51292546  11.51292546]]
    >>> print(kernel.n_dims)
    2
    >>> print(kernel.theta)
    [1.09861229 0.69314718]
    """

    def __init__(self, kernels):
        self.kernels = kernels

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return dict(kernels=self.kernels)

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return np.hstack([kernel.theta for kernel in self.kernels])

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : array of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        k_dims = self.k1.n_dims
        for i, kernel in enumerate(self.kernels):
            kernel.theta = theta[i * k_dims : (i + 1) * k_dims]

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : array of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        return np.vstack([kernel.bounds for kernel in self.kernels])

    def __call__(self, xp, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Note that this compound kernel returns the results of all simple kernel
        stacked along an additional axis.

        Parameters
        ----------
        X : array-like of shape (*batch_shape, n_samples_X, n_features) or list of object, \
            default=None
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (*batch_shape, n_samples_X, n_features) or list of object, \
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of the
            kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (*batch_shape, n_samples_X, n_samples_Y, n_kernels)
            Kernel k(X, Y)

        K_gradient : ndarray of shape \
                (n_samples_X, n_samples_X, n_dims, n_kernels), optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if eval_gradient:
            K = []
            K_grad = []
            for kernel in self.kernels:
                K_single, K_grad_single = kernel(X, Y, eval_gradient)
                K.append(K_single)
                K_grad.append(K_grad_single[..., np.newaxis])
            return xp.dstack(K), xp.concatenate(K_grad, 3)
        else:
            return xp.dstack([kernel(xp, X, Y, eval_gradient) for kernel in self.kernels])

    def diag(self, xp, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to `np.diag(self(X))`; however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (*batch_shape, n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (*batch_shape, n_samples_X, n_kernels)
            Diagonal of kernel k(X, X)
        """
        return xp.vstack([kernel.diag(xp, X) for kernel in self.kernels]).T


class Exponentiation(kernels.Exponentiation):
    """The Exponentiation kernel takes one base kernel and a scalar parameter
    :math:`p` and combines them via

    .. math::
        k_{exp}(X, Y) = k(X, Y) ^p

    Note that the `__pow__` magic method is overridden, so
    `Exponentiation(RBF(), 2)` is equivalent to using the ** operator
    with `RBF() ** 2`.


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : Kernel
        The base kernel

    exponent : float
        The exponent for the base kernel


    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import (RationalQuadratic,
    ...            Exponentiation)
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = Exponentiation(RationalQuadratic(), exponent=2)
    >>> gpr = GaussianProcessRegressor(kernel=kernel, alpha=5,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.419...
    >>> gpr.predict(X[:1,:], return_std=True)
    (array([635.5...]), array([0.559...]))
    """

    def __call__(self, xp, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (*batch_shape, n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_Y, n_features) or list of object,\
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (*batch_shape, n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (*batch_shape, n_samples_X, n_samples_X, n_dims),\
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if eval_gradient:
            K, K_gradient = self.kernel(xp, X, Y, eval_gradient=True)
            K_gradient *= self.exponent * K[:, :, np.newaxis] ** (self.exponent - 1)
            return K**self.exponent, K_gradient
        else:
            K = self.kernel(xp, X, Y, eval_gradient=False)
            return K**self.exponent

    def diag(self, xp, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (*batch_shape, n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (*batch_shape, n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.kernel.diag(xp, X) ** self.exponent


class ConstantKernel(BatchedKernel, kernels.ConstantKernel):
    """Constant kernel.

    Can be used as part of a product-kernel where it scales the magnitude of
    the other factor (kernel) or as part of a sum-kernel, where it modifies
    the mean of the Gaussian process.

    .. math::
        k(x_1, x_2) = constant\\_value \\;\\forall\\; x_1, x_2

    Adding a constant kernel is equivalent to adding a constant::

            kernel = RBF() + ConstantKernel(constant_value=2)

    is the same as::

            kernel = RBF() + 2


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    constant_value : float, default=1.0
        The constant value which defines the covariance:
        k(x_1, x_2) = constant_value

    constant_value_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on `constant_value`.
        If set to "fixed", `constant_value` cannot be changed during
        hyperparameter tuning.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = RBF() + ConstantKernel(constant_value=2)
    >>> gpr = GaussianProcessRegressor(kernel=kernel, alpha=5,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3696...
    >>> gpr.predict(X[:1,:], return_std=True)
    (array([606.1...]), array([0.24...]))
    """

    def __call__(self, xp, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (*batch_shape, n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (*batch_shape, n_samples_X, n_features) or list of object, \
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (*batch_shape, n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (*batch_shape, n_samples_X, n_samples_X, n_dims), \
            optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if X.shape[:-2] != Y.shape[:-2]:
            raise ValueError("X and Y must have the same batch dimensions.")
        
        K = xp.full(
            (*X.shape[:-1], Y.shape[-2]),
            self.constant_value,
            dtype=X.dtype
        )
        if eval_gradient:
            if not self.hyperparameter_constant_value.fixed:
                return (
                    K,
                    xp.full(
                        (*X.shape[:-1], X.shape[-2], 1),
                        self.constant_value,
                        dtype=X.dtype
                    ),
                )
            else:
                return K, xp.empty((*X.shape[:-1], X.shape[:-2], 0), dtype=X.dtype)
        else:
            return K

    def diag(self, xp, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (*batch_shape, n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (*batch_shape, n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return xp.full(X.shape[:-1], self.constant_value, dtype=X.dtype)


class WhiteKernel(BatchedKernel, kernels.WhiteKernel):
    """White kernel.

    The main use-case of this kernel is as part of a sum-kernel where it
    explains the noise of the signal as independently and identically
    normally-distributed. The parameter noise_level equals the variance of this
    noise.

    .. math::
        k(x_1, x_2) = noise\\_level \\text{ if } x_i == x_j \\text{ else } 0


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    noise_level : float, default=1.0
        Parameter controlling the noise level (variance)

    noise_level_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'noise_level'.
        If set to "fixed", 'noise_level' cannot be changed during
        hyperparameter tuning.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel(noise_level=0.5)
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3680...
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([653.0..., 592.1... ]), array([316.6..., 316.6...]))
    """

    def __call__(self, xp, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (*batch_shape, n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (*batch_shape, n_samples_X, n_features) or list of object,\
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (*batch_shape, n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (*batch_shape, n_samples_X, n_samples_X, n_dims),\
            optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = self.noise_level * xp.tile(xp.eye(X.shape[-2], dtype=X.dtype), (*X.shape[:-2],  1, 1))
            if eval_gradient:
                if not self.hyperparameter_noise_level.fixed:
                    return (
                        K,
                        xp.asarray(K[..., None], copy=True),
                    )
                else:
                    return K, xp.empty((*X.shape[:-1], X.shape[-2], 0), dtype=K.dtype)
            else:
                return K
        else:
            if X.shape[:-2] != Y.shape[:-2]:
                raise ValueError("X and Y must have the same batch dimensions.")
            return xp.zeros((*X.shape[:-1], Y.shape[-2]), dtype=X.dtype)

    def diag(self, xp, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (*batch_shape, n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (*batch_shape, n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return xp.full(X.shape[:-1], self.noise_level, dtype=X.dtype)


class RBF(BatchedKernel, kernels.RBF):
    """Radial basis function kernel (aka squared-exponential kernel).

    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length scale
    parameter :math:`l>0`, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel). The kernel is given by:

    .. math::
        k(x_i, x_j) = \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)

    where :math:`l` is the length scale of the kernel and
    :math:`d(\\cdot,\\cdot)` is the Euclidean distance.
    For advice on how to set the length scale parameter, see e.g. [1]_.

    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.
    See [2]_, Chapter 4, Section 4.2, for further details of the RBF kernel.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    References
    ----------
    .. [1] `David Duvenaud (2014). "The Kernel Cookbook:
        Advice on Covariance functions".
        <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_

    .. [2] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * RBF(1.0)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9866...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8354..., 0.03228..., 0.1322...],
           [0.7906..., 0.0652..., 0.1441...]])
    """
    
    def __call__(self, xp, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (*batch_shape, n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (*batch_shape, n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (*batch_shape, n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        # X = xp.atleast_2d(X)
        length_scale = xp.asarray(_check_length_scale(X, self.length_scale), dtype=X.dtype)
        X = X / length_scale
        if Y is None:
            if not eval_gradient or not self.anisotropic or length_scale.shape[0] == 1:
                # In this case we don't need the full distance matrix, we can reduce over the features dimension
                dists = xp.cdist(X, X) ** 2
                K = xp.exp(-0.5 * dists)
            else:
                dists = ((X[..., None, :]) - X[..., None, :, :]) ** 2
                K = xp.exp(-0.5 * dists.sum(-1))
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            Y = Y / length_scale
            dists = xp.cdist(X, Y) ** 2
            K = xp.exp(-0.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, xp.empty((X.shape[0], X.shape[0], 0), dtype=K.dtype)
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * dists)[..., None]
                return K, K_gradient
            elif self.anisotropic:
                K_gradient = K[..., None] * dists
                return K, K_gradient
        else:
            return K
        
    def diag(self, xp, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : ndarray of shape (*batch_shape, n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : ndarray of shape (*batch_shape, n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return xp.ones(X.shape[:-1], dtype=X.dtype)
        

class Matern(RBF, kernels.Matern):
    """Matern kernel.

    The class of Matern kernels is a generalization of the :class:`RBF`.
    It has an additional parameter :math:`\\nu` which controls the
    smoothness of the resulting function. The smaller :math:`\\nu`,
    the less smooth the approximated function is.
    As :math:`\\nu\\rightarrow\\infty`, the kernel becomes equivalent to
    the :class:`RBF` kernel. When :math:`\\nu = 1/2`, the Matérn kernel
    becomes identical to the absolute exponential kernel.
    Important intermediate values are
    :math:`\\nu=1.5` (once differentiable functions)
    and :math:`\\nu=2.5` (twice differentiable functions).

    The kernel is given by:

    .. math::
         k(x_i, x_j) =  \\frac{1}{\\Gamma(\\nu)2^{\\nu-1}}\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )
         \\Bigg)^\\nu K_\\nu\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )\\Bigg)



    where :math:`d(\\cdot,\\cdot)` is the Euclidean distance,
    :math:`K_{\\nu}(\\cdot)` is a modified Bessel function and
    :math:`\\Gamma(\\cdot)` is the gamma function.
    See [1]_, Chapter 4, Section 4.2, for details regarding the different
    variants of the Matern kernel.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    nu : float, default=1.5
        The parameter nu controlling the smoothness of the learned function.
        The smaller nu, the less smooth the approximated function is.
        For nu=inf, the kernel becomes equivalent to the RBF kernel and for
        nu=0.5 to the absolute exponential kernel. Important intermediate
        values are nu=1.5 (once differentiable functions) and nu=2.5
        (twice differentiable functions). Note that values of nu not in
        [0.5, 1.5, 2.5, inf] incur a considerably higher computational cost
        (appr. 10 times higher) since they require to evaluate the modified
        Bessel function. Furthermore, in contrast to l, nu is kept fixed to
        its initial value and not optimized.

    References
    ----------
    .. [1] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import Matern
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9866...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8513..., 0.0368..., 0.1117...],
            [0.8086..., 0.0693..., 0.1220...]])
    """

    def __call__(self, xp, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (*batch_shape, n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (*batch_shape, n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (*batch_shape, n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        
        # X = xp.atleast_2d(X)
        length_scale = xp.asarray(kernels._check_length_scale(X, self.length_scale), dtype=X.dtype)
        full_dists = None
        should_make_squareform = False
        if Y is None:
            # dists = xp.nn.functional.pdist(X / length_scale, p=2)
            if eval_gradient and self.anisotropic:
                # Full distances will be needed afterwards
                full_dists = (X[..., None, :] - X[..., None, :, :]) ** 2 / (length_scale**2)

                def get_dists():
                    return xp.sqrt(full_dists.sum(-1))
                
                def get_sqdists():
                    return full_dists.sum(-1)
                
            else:
                tmp = X / length_scale
                dists = xp.cdist(tmp, tmp)
                del tmp

                def get_dists():
                    return dists
                
                def get_sqdists():
                    return dists ** 2
            
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = xp.cdist(X / length_scale, Y / length_scale)

            def get_dists():
                return dists
            
            def get_sqdists():
                return dists ** 2

        if self.nu == 0.5:
            K = xp.exp(-get_dists())
        elif self.nu == 1.5:
            K = get_dists() * math.sqrt(3)
            K = (1.0 + K) * xp.exp(-K)
        elif self.nu == 2.5:
            K = get_dists() * math.sqrt(5)
            K = (1.0 + K + K**2 / 3.0) * xp.exp(-K)
        elif self.nu == np.inf:
            K = xp.exp(-get_sqdists() / 2.0)
        else:  # general case; expensive to evaluate
            raise NotImplementedError()
            # K = dists
            # K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            # tmp = math.sqrt(2 * self.nu) * K
            # K.fill((2 ** (1.0 - self.nu)) / gamma(self.nu))
            # K *= tmp**self.nu
            # K *= kv(self.nu, tmp)

        if should_make_squareform:
            K = _squareform(xp, K, X.shape[-2])

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                K_gradient = xp.empty((X.shape[0], X.shape[0], 0), dtype=K.dtype)
                return K, K_gradient

            # We need to recompute the pairwise dimension-wise distances
            if self.anisotropic:
                D = full_dists
            else:
                # D = squareform(dists**2)[:, :, None]
                D = get_sqdists()[..., None]

            if self.nu == 0.5:
                denominator = xp.sqrt(D.sum(axis=-1))
                denominator[denominator == 0] = 1  # Prevent division by zero
                K_gradient = K[..., None] * (D / denominator[..., None])
                del denominator
            elif self.nu == 1.5:
                K_gradient = 3 * D * xp.exp(-xp.sqrt(3 * D.sum(-1)))[..., None]
            elif self.nu == 2.5:
                tmp = xp.sqrt(5 * D.sum(-1))[..., None]
                K_gradient = 5.0 / 3.0 * D * (tmp + 1) * xp.exp(-tmp)
                del tmp
            elif self.nu == np.inf:
                K_gradient = D * K[..., None]
            else:
                raise NotImplementedError()
                # approximate gradient numerically
                # def f(theta):  # helper function
                #     return self.clone_with_theta(theta)(X, Y)

                # return K, _approx_fprime(self.theta, f, 1e-10)

            if not self.anisotropic:
                return K, K_gradient.sum(-1)[..., None]
            else:
                return K, K_gradient
        else:
            return K


# TODO: implement remainign kernels, mainly RationalQuadratic, ExpSine, DotProduct, PairWiseK


def _squareform(xp, x, n):
    """Convert a 1D array to a square matrix."""
    out = xp.empty((*x.shape[:-1], n, n), dtype=x.dtype)
    ind = np.tril_indices(n, k=-1)
    assert ind[0].size * np.cumprod(out.shape[:-2]) == x.size, "Unexpected error. Contact the developer"
    out[..., ind[0], ind[1]] = x
    del ind
    out += out.transpose(-2, -1)
    return out


def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[-1] != length_scale.shape[0]:
        raise ValueError(
            "Anisotropic kernel must have the same number of "
            "dimensions as data (%d!=%d)" % (length_scale.shape[0], X.shape[1])
        )
    return length_scale
