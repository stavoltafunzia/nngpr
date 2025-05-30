""" This exmaple fits a simple TorchNngpr model on the california housing dataset.
Cuda GPU is used if available """
import torch

import numpy as np
from sklearn import datasets

from nngpr.torch_nngpr import TorchNngpr
from nngpr import batched_kernels


if __name__ == '__main__':

    # Fetch data
    print("Loading data")
    data = datasets.fetch_california_housing()
    x = data['data']
    y = data['target']
    x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
    y = (y - np.mean(y)) / np.std(y)

    # Train
    print("Fitting")
    kernel = batched_kernels.WhiteKernel() + batched_kernels.ConstantKernel() * batched_kernels.RBF(np.ones(x.shape[1]))
    mdl = TorchNngpr('cuda' if torch.cuda.is_available() else 'cpu', num_nn=64)
    mdl.fit(x, y)

    # Predict
    print("Predicting")
    y_pred = mdl.predict(x)

    print("All done")
