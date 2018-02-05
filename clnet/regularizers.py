#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from clnet.tensor import Tensor

class Regularizer:

    def loss(self, params: Tensor) -> float:
        raise NotImplementedError


class L2(Regularizer):

    def __init__(self, strength: float = 1e-3) -> None:
        self.strength = strength

    def loss(self, params: Tensor) -> float:

        return 0.5 * self.strength * np.sum(np.square(params))
