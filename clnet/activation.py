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
    grads[grads <= 0] = 0
    return grads

def sigmoid(inputs: Tensor) -> Tensor:

    return 1 / (1 + np.exp(- inputs))

def sigmoid_prime(grads: Tensor ) -> Tensor:
    return grads * (1 - grads)

def softmax(inputs: Tensor) -> Tensor:

    # stable softmax that does not run into inf / overflow
    exps = np.exp(inputs - np.max(inputs))

    if inputs.ndim == 1:
        sum_axis = 0
    else:
        sum_axis = 1

    return exps / np.sum(exps, axis=sum_axis, keepdims=True)

def softmax_prime(grads: Tensor) -> Tensor:
    """
    Assuming softmax at the "end" of the network, and followed
    by cross-entropy loss.
    """
    return grads

def tanh(inputs: Tensor) -> Tensor:

    return np.tanh(inputs)

def tanh_prime(grads: Tensor) -> Tensor:
    """
    Derivative of tanh(x) is 1 - tanh ** 2 (x)
    """
    return 1 - np.tanh(grads) ** 2
