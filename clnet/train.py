#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Most of the code taken from:
# https://github.com/joelgrus/joelnet

"""
Here's a function that can train a neural net
"""

from clnet.tensor import Tensor
from clnet.nn import NeuralNet
from clnet.loss import Loss, SSE
from clnet.optim import Optimizer, SGD
from clnet.data import DataIterator, BatchIterator


def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = SSE(),
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)
