#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Most of the code taken from:
# https://github.com/joelgrus/joelnet

"""
A NeuralNet is just a collection of layers.
It behaves a lot like a layer itself, although
we're not going to make it one.
"""
from typing import Sequence, Iterator, Tuple

from clnet.tensor import Tensor
from clnet.layers import Layer


class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def reg_losses(self) -> Iterator[float]:

        reg_loss = 0
        for layer in self.layers:
            if "W" in layer.params:
                reg_loss += layer.reg_loss
        return reg_loss
