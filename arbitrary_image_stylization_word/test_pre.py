#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
# import argparse
# from PIL import ImageFile
from chainer import cuda, serializers
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


model_path = 'models/out5/final.model'
vgg = VGGNet()
serializers.load_hdf5('VGG.model', vgg)
tinynet = wordQueryNet()
serializers.load_npz(model_path, tinynet)
tinynet.to_gpu()
words = ['布', '植物', 'ガラス', '革', '金属', '紙', 'プラスチック', '石', '水', '木']
for word in words:
    for count in range(16):
        filename = 'images/valid/{}.jpg'.format(count)
        vgg_param = vggparamater(filename, 0, vgg)[0]
        concatted = concatData(word, vgg_param)
        concatted_g = cuda.to_gpu(concatted)
        style_params = tinynet(concatted_g)
        print(style_params.data)
        style_params_cpu = cuda.to_cpu(style_params.data)
        # style_params = np.reshape(style_params, (1, 1, 1, 100))

        print('params/{}_{}.pickle'.format(word, count))
        f = open('params/{}_{}.pickle'.format(word, count), 'w')
        pickle.dump(style_params_cpu, f)
        f.close
