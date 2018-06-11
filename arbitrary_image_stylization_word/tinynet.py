#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
import cupy


# bottoleneckをつけてみる
class wordQueryNet(chainer.Chain):
    def __init__(self):
        heinit = chainer.initializers.HeNormal()
        super(wordQueryNet, self).__init__(
            # c1 = L.Convolution2D(1200,1,1, stride=1, pad=0, initialW=heinit),
            l1=L.Linear(500, 256, initialW=heinit),
            l2=L.Linear(256, 256),
            l3=L.Linear(256, 100),
            b1=L.BatchNormalization(50),
            b2=L.BatchNormalization(50),
        )

# Dropout, BN を入れてみる
# tanhを入れてみる
    def __call__(self, h, train=True):
        # h = Variable(i)
        # print(h.data.shape)
        # print("h")
        # h = self.c1(F.relu(h))
        h = self.l1(h)
        h = F.relu(h)
        # print(h.data[0][0:10])
        h = F.dropout(h, 0.5, train=train)
        h = self.l2(h)
        h = F.relu(h)
        #print(h.data[0][0:10])
        # print(self.l2.W.data[0:10])
        h = F.dropout(h, 0.5, train=train)
        h = self.l3(h)
        # print(h.data[0][0:10])
        # hr = cupy.zeros((1, 1, 1, 100))
        # hr = F.reshape(h, (1, 1, 1, 100))
        return h
