#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Most of the code taken from:
# https://github.com/joelgrus/joelnet

"""
A spiral data set example that cannot be learned with a linear classifier.

Source of the data set:
http://cs231n.github.io/neural-networks-case-study/#grad
"""
import numpy as np

from clnet.train import train
from clnet.nn import NeuralNet
from clnet.layers import Linear, Relu, Softmax
from clnet.loss import CrossEntropy

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes

inputs = np.zeros((N*K,D)) # data matrix (each row = single example)
targets = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  inputs[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  targets[ix] = j

net = NeuralNet([
    Linear(input_size=2, output_size=100),
    Relu(),
    Linear(input_size=100, output_size=3),
    Softmax()
])

train(net, inputs, targets, num_epochs=20, loss=CrossEntropy())

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)
