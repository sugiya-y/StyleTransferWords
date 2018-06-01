import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
import cupy


class wordQueryNet(chainer.Chain):
    def __init__(self):
        heinit = chainer.initializers.HeNormal()
        super(wordQueryNet, self).__init__(
            # c1 = L.Convolution2D(1200,1,1, stride=1, pad=0, initialW=heinit),
            l1=L.Linear(500, 30),
            l2=L.Linear(30, 30),　# bottoleneckをつけてみる
            l3=L.Linear(30, 100),
        )

    def __call__(self, i, test=True):
        h = Variable(i)
        # print(h.data.shape)
        # print("h")
        # h = self.c1(F.relu(h))
        h = self.l1(h)
        h = self.l2(F.relu(h))
        h = self.l3(F.relu(h)) # Dropout, BN を入れてみる
        hr = cupy.zeros((1, 1, 1, 100)) # tanhを入れてみる
        hr = F.reshape(h, (1, 1, 1, 100))
        return hr
