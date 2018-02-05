#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Most of the code taken from:
# https://github.com/joelgrus/joelnet

"""
A loss function measures how good our predictions are,
we can use this to adjust the parameters of our network
"""
import numpy as np

from clnet.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class SSE(Loss):
    """
    Sum squared error loss.
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)

class MSE(Loss):
    """
    Mean squared error loss.
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.mean((predicted - actual) ** 2, axis=-1)


    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class CrossEntropy(Loss):

    def loss(self, predicted: Tensor, actual: Tensor) -> float:

        num_samples = actual.shape[0]

        # for each array of predicted probabilities, pick the probability of
        # the true class
        logs = - np.log(predicted[range(num_samples), actual])

        return np.sum(logs) / num_samples

    def grad(self, predicted: Tensor, actual: Tensor) -> float:

        num_samples = actual.shape[0]

        # subtract the true class label, which happens to be 1
        predicted[range(num_samples), actual] -= 1

        predicted = predicted / num_samples

        return predicted
