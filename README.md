# NNGPR: Nearest Neighbors Gaussian Process Regressor

This project implements the Nearest Neighbor Gaussian Process Regressor (NNGPR) according to [Datta 2016](https://arxiv.org/abs/1406.7343).
In a nutshell, this model works by building a local Gaussian Process around the nearest neighbors of a given point. 

##### Advantages over standard Gaussian Process Regressor
NNGPR overcomes quadratic complexity of the standard Gaussian Processes. The complexity of a NNGPR with M nearest neighbors is N*M^2
for the Gaussian Process part (kernel and matrix operations, usually the bottleneck), and N^2 for the nearest neighbors search. 
Moreover NNGPR does not have a quadratic memory usage since it never stores the full kernel or covariance matrix, thus allows to use GPs on large datasets.


##### Python requirements

- scikit-learn (tested on 1.6.1)

Optional requirements to run the benchmark notebook:

- matplotlib
- ucimlrepo


##### Run unittests

`python -m test.test_nngpr`

##### Usage exmaple

```
from sklearn import datasets
from sklearn.gaussian_process import kernels

from nngpr import NNGPR

data = datasets.fetch_california_housing()
x = data['data']
y = data['target']

kernel = kernels.ConstantKernel(1) * kernels.RBF(length_scale=1) + kernels.WhiteKernel(1)
mdl = NNGPR(n_jobs=-1)
mdl.fit(x, y)
mdl.predict(x)
```
