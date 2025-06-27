# Nngpr: Nearest Neighbors Gaussian Process Regressor

This project implements the Nearest Neighbor Gaussian Process Regressor (Nngpr) according to [Datta 2016](https://arxiv.org/abs/1406.7343).
This model works by building a local Gaussian Process around the kernel-space nearest neighbors of a given point. 
Since the local Gaussian Process depends only a small number of nearest neighbors, this model  overcomes the quadratic complexity of the standard Gaussian Processes regressor.
The complexity of Nngpr with n points and m nearest neighbors is n*m^2 for the Gaussian Process kernel and matrix operations, and n^2 for the nearest neighbors search.
For large datasets the nearest neighbors search becomes the main bottleneck, however it can be accelerated by GPUs.
Nngpr does not have a quadratic memory footprint since it never stores the full kernel or covariance matrices, thus can be used on datasets much larger than what typically possible with traditional Gaussian Processes.

Some very basic benchmarks are provided in the notebook `benchmark.ipynb`. 


## Installation

Nngpr can be installed from pip with:

> pip install nngpr

By default, this command does not install gpu-accellerated Nngpr. To install the cupy or torch accellerated Nngpr (see Usage section below), simply do:

> pip install nngpr[torch]

or, if you don't want to install torch, you can rely on cupy:

> pip install nngpr[cupy]

## Usage

Nearest Neighbor Gaussian Process Regressor is implementend in a scikit-learn-like interface. 
Nngpr must be used with kernels that inherits from `nngpr.batched_kernels.BatchedKernel`. The package `nngpr.batched_kernels` already provides a sckit-learn-like implementation of the most common kernels.

There are three different implementations of Nngpr:

- Numpy-based (NumpyNngpr): this is the simplest implementation and requires only scikit-learn as dependency. It runs on CPU and is not practically usable for large datasets as it would turn out to be rather slow (see `benchmark.ipynb` for some rough numbers). This is always available within the `nngpr` package.
- Torch-based (TorchNngpr): this implementation uses `torch` and can potentially run on any torch device. On Cuda devices, kernels of the type WhiteKernel + ConstantKernel * RBF gets accellerated by custom Cuda kernels (here kernels refers to Cuda software) through cupy. This implementation is available when installing `nngpr[torch]`.
- Cupy-based (CupyNngpr): this implementation uses `cupy` to run on Nvidia GPUs. It does not require `torch`, however it currently supports only kernels of the type WhiteKernel + ConstantKernel * RBF. This implementation is available when installing `nngpr[cupy]` or `nngpr[torch]`.


CupyNngpr and TorchNngpr have practically the same performance on Nvidia GPUs, with the latter supporting a broader family of kernels. NumpyNngpr performs better than cpu-based TorchNngpr.
For specific examples on using Nngpr please see the `examples` folder.


## Run unittests

Some unittest are included in the `nngpr` to check that everything works correctly. You can run them with:

- `python -m nngpr.run_tests`

## Python requirements

Requirements are automatically installed by `pip`. If you want to run the benchmark notebook, install these additional packages:

- matplotlib
- ucimlrepo
- gpytorch (if you want to run GPyTorch models, tested on gpytorch v1.14)

