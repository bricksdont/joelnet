#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Most of the code taken from:
# https://github.com/joelgrus/joelnet

"""
Our neural nets will be made up of layers.
Each layer needs to pass its inputs forward
and propagate gradients backward. For example,
a neural net might look like

inputs -> Linear -> Tanh -> Linear -> output
"""
from typing import Dict, Callable

import numpy as np

from clnet.tensor import Tensor
from clnet.regularizers import Regularizer
from clnet import activation as A

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError


class Linear(Layer):
    """
    computes output = inputs @ w + b
    """
    def __init__(self, input_size: int,
                 output_size: int,
                 regularizer: Regularizer = None) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

        self.reg_loss = 0
        self.regularizer = regularizer

    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b

        Also keep track of regularization loss in forward pass
        """
        self.inputs = inputs

        if self.regularizer:
            self.reg_loss = self.regularizer.loss(self.params["w"])

        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad

        if self.regularizer:
            self.grads["w"] += self.regularizer.strength * self.params["w"]

        return grad @ self.params["w"].T


F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    An activation layer just applies a function
    elementwise to its inputs
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad


class Tanh(Activation):
    def __init__(self):
        super().__init__(A.tanh, A.tanh_prime)

class Relu(Activation):
    def __init__(self):
        super().__init__(A.relu, A.relu_prime)

    def backward(self, grad: Tensor) -> Tensor:

        outputs = self.f(self.inputs)
        grad[outputs <= 0] = 0
        return grad

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(A.sigmoid, A.sigmoid_prime)

class Softmax(Activation):
    def __init__(self):
        super().__init__(A.softmax, A.softmax_prime)
