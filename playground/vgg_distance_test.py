#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
# import argparse
# from PIL import ImageFile
from chainer import cuda, serializers, Variable
from tinynet import wordQueryNet
from tinynet_novgg import wordQueryNetNoVGG
from wordparam import word2vector
from vggparam import vggparamater
from vggnet import VGGNet
import pickle
import argparse
import cv2 as cv
# import chainer
# import chainer.functions as F


vgg = VGGNet()
serializers.load_hdf5('/tmp/VGG.model', vgg)
mean = np.array([103.939, 116.779, 123.68])

img = cv.imread(image).astype(np.float32)
img -= mean
img = cv.resize(img, (224, 224)).transpose((2, 0, 1))
img = img[np.newaxis, :, :, :]

gpu = 0
if gpu >= 0:
    cuda.get_device(gpu).use()
    vgg.to_gpu()
    img = cuda.cupy.asarray(img, dtype=np.float32)

pred = vgg(Variable(img), None)
