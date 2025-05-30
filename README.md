# Nngpr: Nearest Neighbors Gaussian Process Regressor

This project implements the Nearest Neighbor Gaussian Process Regressor (Nngpr) according to [Datta 2016](https://arxiv.org/abs/1406.7343).
This model works by building a local Gaussian Process around the kernel-space nearest neighbors of a given point. 
Since the local Gaussian Process depends only a small number of nearest neighbors, this model  overcomes the quadratic complexity of the standard Gaussian Processes regressor.
The complexity of Nngpr with n points and m nearest neighbors is n*m^2 for the Gaussian Process kernel and matrix operations, and n^2 for the nearest neighbors search.
For large datasets the nearest neighbors search becomes the main bottleneck, however it can be accelerated by GPUs.
Nngpr does not have a quadratic memory footprint since it never stores the full kernel or covariance matrices, thus can be used on datasets much larger than what typically possible with traditional Gaussian Processes.

Some very basic benchmarks are provided in the notebook `benchmark.ipynb`. 

#### Python requirements

- scikit-learn (tested on v1.6.1)

Requirements to use CudaNngpr:

- cupy (with cusolver and cublas, tested on cupy-cuda12x v13.4.1)

Requirements to use TorchNngpr:

- torch (tested on v2.7.0)
- cupy is highly suggested if cuda devices are being used

Optional requirements to run the benchmark notebook:

- matplotlib
- ucimlrepo
- gpytorch (if you want to run GPyTorch models, tested on gpytorch v1.14)


#### Usage

Nearest Neighbor Gaussian Process Regressor is implementend with a scikit-learn-like interface. 
Nngpr must be used with kernels that inherits from `nngpr.batched_kernels.BatchedKernel`; the package `nngpr.batched_kernels` already provides a sckit-learn-like implementation of the most common kernels.

There are three different implementations of Nngpr:

- Numpy-based (NumpyNngpr): this is the simplest implementation and requires only scikit-learn as dependency. It runs on CPU and is not practically usable for large datasets as it would turn out to be rather slow (see `benchmark.ipynb` for some rough numbers).
- Cupy-based (CudaNngpr): this implementation uses `cupy` to run on Nvidia GPUs.
- Torch-based (TorchNngpr): this implementation uses `pytorch` and can potentially run on any pytorch device. Kernels of the type WhiteKernel + ConstantKernel * RBF can benefit from substantial speed-up if cupy is also installed, as custom Cuda kernels (here kernels refers to Cuda software) can be exploited.

CudaNngpr and TorchNngpr have practically the same performance on Cuda GPUs. NumpyNngpr performs better than cpu-based TorchNngpr.

For specific examples on using Nngpr please see the `examples` folder.


##### Run unittests

Some unittest are included to check that everything works:

- `python -m test.test_batched_kernels`
- `python -m test.test_numpy_nngpr` (make take a few minutes to run)
- `python -m test.test_cuda_nngpr` (run only if you have `cupy` installed)
- `python -m test.test_torch_nngpr` (run only if you have `torch` installed)
