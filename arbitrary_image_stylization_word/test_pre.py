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
    vec= vec / np.linalg.norm(vec)
    vec = vec * 5
    # print('word: ' + str(np.mean(vec)))
    # print('vgg: ' + str(np.mean(vgg_img_param)))
    vgg_img_param=np.array(vgg_img_param)
    vgg_img_param= vgg_img_param/ np.linalg.norm(vgg_img_param)
    concated = np.concatenate((vec, vgg_img_param))
    param = np.reshape(concated, (500, 1))

    return np.transpose(param)


def w2v(word):
    # 時間がかかるのでデータがあれば読み込む
    if os.path.isfile('wordparam/word2vecter' + word + '.npy'):
        vec = np.load('wordparam/word2vecter' + word + '.npy')
    else:
        vec = word2vector(word)
        # 時間がかかるのでデータを保存する
        np.save('wordparam/word2vecter' + word + '.npy', vec)

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
model_path = 'models/{}/final.model'.format(model)
vgg = VGGNet()
serializers.load_hdf5('/tmp/VGG.model', vgg)
if args.usevgg == 1:
    tinynet = wordQueryNet()
else:
    tinynet = wordQueryNetNoVGG()
serializers.load_npz(model_path, tinynet)
tinynet.to_gpu()
words = ['布', '植物', 'ガラス', '革', '金属', '紙', 'プラスチック', '石', '水', '木', '樹脂', 'アクリル', 'アルミニウム', '牛皮', 'レンガ', '絹']
for word in words:
    for count in range(36):
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
        f = open('params/{}_{}.pickle'.format(word, count), 'w')
        pickle.dump(style_params_cpu, f)
        f.close
