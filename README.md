# NNGPR: Nearest Neighbors Gaussian Process Regressor

This project implements the Nearest Neighbor Gaussian Process Regressor (NNGPR) according to [Datta 2016](https://arxiv.org/abs/1406.7343).
In a nutshell, this model works by building a local Gaussian Process around the nearest neighbors of a given point. 
Cuda can be used to speed-up the model significantly. Using scikit-learn class names, currently only kernels of the type WhiteKernel + ConstantKernel * RBF are supported in the Cuda-based implementation.


##### Advantages over standard Gaussian Process Regressor

NNGPR overcomes quadratic complexity of the standard Gaussian Processes. The complexity of a NNGPR with M nearest neighbors is N*M^2
for the Gaussian Process part (kernel and matrix operations, usually the bottleneck), and N^2 for the nearest neighbors search. 
Moreover NNGPR does not have a quadratic memory usage since it never stores the full kernel or covariance matrix, thus allows to use GPs on large datasets.


##### Python requirements

- scikit-learn (tested on 1.6.1)

Requirements to use the Cuda-based NNGPR (CUNNGPR):

- cupy (with cusolver and cublas)

Optional requirements to run the benchmark notebook:

- matplotlib
- ucimlrepo
- gpytorch (if you want to run VNNGP)


##### Usage exmaple

Nearest Neighbor Gaussian Process Regressor is implementend in two classes: `nngpr.nngpr.NNGPR` runs on CPU, while `nngpr.cunngpr.CUNNGPR` runs on GPU through `cupy`.
Both classes exposes a scikit-learn-like interface, and scikit-learn kernels should be used. 


```
from sklearn import datasets
from sklearn.gaussian_process import kernels

#from nngpr import NNGPR  # Use this if you want to use the CPU-only implementation
from nngpr.cunngpr import CUNNGPR

data = datasets.fetch_california_housing()
x = data['data']
y = data['target']

kernel = kernels.ConstantKernel(1) * kernels.RBF(1) + kernels.WhiteKernel(1)
mdl = CUNNGPR(allow_downcast_f32='only-nn')  # Use mdl = NNGPR(n_jobs=-1) if you want to use the CPU-only implementation
mdl.fit(x, y)
mdl.predict(x)
```

##### Run unittests

`python -m test.test_nngpr`

or, to test the Cuda-based implementation

`python -m test.test_cunngpr`
