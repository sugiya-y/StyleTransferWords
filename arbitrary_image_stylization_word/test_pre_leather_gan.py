#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding:utf-8

import os
import numpy as np
# import argparse
# from PIL import ImageFile
from chainer import cuda, serializers, Variable
from gensim.models import word2vec
from vggparam import vggparamater
from vggnet import VGGNet
import pickle
import argparse
from tinygan import Generator
# import chainer
# import chainer.functions as F


wmodel = word2vec.Word2Vec.load("../word2vec_en/enwiki.model")


def concatData(word, vgg_img_param):
    vec = wmodel.wv[word]

    param = np.zeros((500, 1))
    vec = vec / np.linalg.norm(vec)
    # vec = vec * 2
    # print('word: ' + str(np.mean(vec)))
    # print('vgg: ' + str(np.mean(vgg_img_param)))
    vgg_img_param = np.array(vgg_img_param)
    vgg_img_param = vgg_img_param / np.linalg.norm(vgg_img_param)
    concated = np.concatenate((vec, vgg_img_param))
    param = np.reshape(concated, (500, 1))

    return np.transpose(param)


def w2v(word):
    vec = wmodel.wv[word]
    vec = vec / np.linalg.norm(vec)

    return vec


parser = argparse.ArgumentParser(
    description='Real-time style transfer image generator')
parser.add_argument('--model', '-m', type=str, required=True,
                    help='model path')
parser.add_argument('--usevgg', '-u', default=1, type=int,
                    help='use or dont use vgg: 0 or 1')
args = parser.parse_args()

model = args.model
model_path = 'models/leather_gan/epoch_{}.model'.format(model)
vgg = VGGNet()
serializers.load_hdf5('/tmp/VGG.model', vgg)
if args.usevgg == 1:
    tinynet = Generator()
else:
    tinynet = Generator()
serializers.load_npz(model_path, tinynet)
tinynet.to_gpu()
words = ['recent',
        'fresh',
        'advanced',
        'current',
        'late',
        'modern',
        'elderly',
        'gray',
        'mature',
        'tired',
        'fossil',
        'senior',
        'veteran',
        'broken',
        'new',
        'old',
        ]
for word in words:
    for count in range(28):
        filename = 'images/valid/{}.jpg'.format(count)
        vgg_param = vggparamater(filename, 0, vgg)[0]
        if args.usevgg == 1:
            concatted = concatData(word, vgg_param)
        else:
            concatted = w2v(word)
        print('moto:')
        # print(concatted[0][400:420])
        # print(concatted[0][0:10])
        if args.usevgg == 0:
            concatted = np.reshape(concatted, (1, 200))
        print(concatted.shape)
        concatted_g = Variable(cuda.to_gpu(concatted))
        style_params = tinynet(concatted_g, train=False)
        # print(style_params.data.shape,concatted_g.data.shape)
        print('params:')
        print(style_params.data[0][0:10])
        style_params_cpu = cuda.to_cpu(style_params.data)
        style_params_cpu = np.reshape(style_params_cpu, (1, 1, 1, 100))

        print('params/{}_{}.pickle'.format(word, count))
        f = open('params/{}_{}.pickle'.format(word, count), 'wb')
        pickle.dump(style_params_cpu, f)
        f.close
