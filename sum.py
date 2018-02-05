#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Müller / mmueller@cl.uzh.ch

"""
Learn a function that can sum inputs.
"""
import numpy as np
import sys

from clnet.train import train
from clnet.nn import NeuralNet
from clnet.layers import Linear, Tanh, Relu, Sigmoid


def test_model(net: NeuralNet, inputs: np.ndarray, targets: np.ndarray) -> None:
    for x, y in zip(inputs, targets):
        predicted = net.forward(x)
        print(x, predicted, y)

num_examples = 10
numbers_per_example = 10


inputs = np.random.randint(10, size=(num_examples, numbers_per_example)) * 0.1

targets = np.array([[np.sum(example)] for example in inputs])

net = NeuralNet([
    Linear(input_size=10, output_size=10),
    Tanh(),
    Linear(input_size=10, output_size=1)
])

test_model(net, inputs, targets)

train(net, inputs, targets, num_epochs=1000)

test_model(net, inputs, targets)
