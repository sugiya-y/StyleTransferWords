#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
# import argparse
# from PIL import ImageFile
from chainer import cuda, serializers, Variable
from tinynet import wordQueryNet
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


parser = argparse.ArgumentParser(
    description='Real-time style transfer image generator')
parser.add_argument('--model', '-m', type=str, required=True,
                    help='model path')
args = parser.parse_args()

model = args.model
model_path = 'models/{}/final.model'.format(model)
vgg = VGGNet()
serializers.load_hdf5('/tmp/VGG.model', vgg)
tinynet = wordQueryNet()
serializers.load_npz(model_path, tinynet)
tinynet.to_gpu()
words = ['布', '植物', 'ガラス', '革', '金属', '紙', 'プラスチック', '石', '水', '木', '樹脂', 'アクリル', 'アルミニウム', '牛皮', 'レンガ', '絹']
for word in words:
    for count in range(36):
        filename = 'images/valid/{}.jpg'.format(count)
        vgg_param = vggparamater(filename, 0, vgg)[0]
        concatted = concatData(word, vgg_param)
        print('moto:')
        print(concatted[0][400:420])
        print(concatted[0][0:10])
        concatted_g = Variable(cuda.to_gpu(concatted))
        style_params = tinynet(concatted_g, train=False)
        #print(style_params.data.shape,concatted_g.data.shape)
        print('params:')
        print(style_params.data[0][0:10])
        style_params_cpu = cuda.to_cpu(style_params.data)
        style_params_cpu = np.reshape(style_params_cpu, (1, 1, 1, 100))

        print('params/{}_{}.pickle'.format(word, count))
        f = open('params/{}_{}.pickle'.format(word, count), 'w')
        pickle.dump(style_params_cpu, f)
        f.close
