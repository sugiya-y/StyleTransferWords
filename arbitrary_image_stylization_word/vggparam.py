#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import cPickle as pickle
from chainer import cuda
from chainer import Variable


def vggparamater(image, gpu, vgg):  # image:str(path to image file) gpu:number(gpu-id)

    mean = np.array([103.939, 116.779, 123.68])
    img = cv.imread(image).astype(np.float32)
    img -= mean
    img = cv.resize(img, (224, 224)).transpose((2, 0, 1))
    img = img[np.newaxis, :, :, :]

    if gpu >= 0:
        cuda.get_device(gpu).use()
        vgg.to_gpu()
        img = cuda.cupy.asarray(img, dtype=np.float32)

    pred = vgg(Variable(img), None)

    if gpu >= 0:
        pred = cuda.to_cpu(pred.data)
    else:
        pred = pred.data

    with open('pca.pickle', 'rb') as f:
        pca = pickle.load(f)

    result = pca.transform(pred)

    # PCAmean = np.load('PCAmean.npy')
    # PCAeigen = np.load('PCAeigen.npy')

    # result = cv2.PCAProject(pred,PCAmean,PCAeigen)

    # print np.shape(result)
    return result
    # return pred
