#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
import cupy


class Generator(chainer.Chain):
    def __init__(self):
        heinit = chainer.initializers.HeNormal()
        super(Generator, self).__init__(
            # c1 = L.Convolution2D(1200,1,1, stride=1, pad=0, initialW=heinit),
            l1=L.Linear(200, 200, initialW=heinit),
            l2=L.Linear(200, 200),
            l3=L.Linear(200, 100),
        )

# Dropout, BN を入れてみる
# tanhを入れてみる
    def __call__(self, h, train=True):
        h = self.l1(h)
        h = F.relu(h)
        h = F.dropout(h, 0.5, train=train)
        h = self.l2(h)
        h = F.relu(h)
        h = F.dropout(h, 0.5, train=train)
        h = self.l3(h)
        return h


class Discriminator(chainer.Chain):
    def __init__(self):
        heinit = chainer.initializers.HeNormal()
        super(Discriminator, self).__init__(
            l1=L.Linear(100, 100, initialW=heinit),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 2),
        )

    def __call__(self, h, train=True):
        h = self.l1(h)
        h = F.relu(h)
        h = F.dropout(h, 0.5, train=train)
        h = self.l2(h)
        h = F.relu(h)
        h = F.dropout(h, 0.5, train=train)
        h = self.l3(h)
        return h
