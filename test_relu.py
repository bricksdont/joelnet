#!/usr/bin/env python
# -*- coding: utf-8 -*-

from clnet import activation

import numpy as np

inputs = np.array([1, 2, 0, 3, 0, -9])

print(activation.sigmoid(inputs))
