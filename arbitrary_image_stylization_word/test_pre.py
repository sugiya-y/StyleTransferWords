#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
# import argparse
# from PIL import ImageFile
from chainer import cuda
from tinynet import wordQueryNet
from wordparam import word2vector
from vggparam import vggparamater
from vggnet import VGGNet
import pickle
# import chainer
# import chainer.functions as F


def concatData(word, vgg_img_param):
    # 時間がかかるのでデータがあれば読み込む
    if os.path.isfile('wordparam/word2vecter' + word + '.npy'):
        vec = np.load('wordparam/word2vecter' + word + '.npy')
    else:
        vec = word2vector(word)
        # 時間がかかるのでデータを保存する
        np.save('wordparam/word2vecter' + word + '.npy', vec)

    param = np.zeros((500, 1))
    vec = vec * 40
    concated = np.concatenate((vec, vgg_img_param))
    param = np.reshape(concated, (500, 1))

    return np.transpose(param)


word = '革'
filename = 'images/style_images/La_forma.jpg'
vgg = VGGNet()
tinynet = wordQueryNet()
tinynet.to_gpu()
vgg_param = vggparamater(filename, 0, vgg)[0]
concatted = concatData(word, vgg_param, )
concatted_g = cuda.to_gpu(concatted)
style_params = tinynet(concatted_g)
style_params_cpu = cuda.to_cpu(style_params.data)
# style_params = np.reshape(style_params, (1, 1, 1, 100))

print(style_params_cpu)
f = open('params/pre.pickle', 'w')
pickle.dump(style_params_cpu, f)
f.close
