#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Most of the code taken from:
# https://github.com/joelgrus/joelnet

import numpy as np

from clnet.tensor import Tensor


def linear(inputs: Tensor) -> Tensor:

    return inputs

def linear_prime(grads: Tensor) -> Tensor:

    return grads

def relu(inputs: Tensor) -> Tensor:
    return np.maximum(inputs, 0)

def relu_prime(grads: Tensor) -> Tensor:
    """
    Derivative is 1 if the forward pass number is
    positive, and 0 otherwise.
    """
    return (grads > 0) * 1

def sigmoid(inputs: Tensor) -> Tensor:

    return 1 / (1 + np.exp(- inputs))

def sigmoid_prime(grads: Tensor ) -> Tensor:
    return grads * (1 - grads)

def softmax(inputs: Tensor) -> Tensor:

    # stable softmax that does not run into inf / overflow
    exps = np.exp(inputs - np.max(inputs))

    return exps / exps.sum(axis=0)

def softmax_prime(grads: Tensor) -> Tensor:

    raise NotImplementedError

def tanh(inputs: Tensor) -> Tensor:

    return np.tanh(inputs)

def tanh_prime(grads: Tensor) -> Tensor:
    """
    Derivative of tanh(x) is 1 - tanh ** 2 (x)
    """
    return 1 - np.tanh(grads) ** 2

batch = [[5, 2, 3], [0.3, 2, 1]]

print(softmax(batch))
