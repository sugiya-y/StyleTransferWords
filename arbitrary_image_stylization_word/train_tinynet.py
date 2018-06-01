#!/usr/bin/env python
# -*- coding: utf-8 -*-

# まずは画像のパラメータを拾得する
from generate_style_parameter import styleParam
import numpy as np
import time

start = time.time()
filenames = np.load('images/filenames.npy')
print('content image loaded!')
print('preprocessing target style data')
filenames = filenames[0:12]  # for testing
target_img_param = styleParam(filenames)
calctime = time.time() - start
print('inception v3 time: ' + str(calctime) + '[sec]')

import os
import argparse
from PIL import ImageFile
from chainer import cuda, Variable, optimizers, serializers
from tinynet import wordQueryNet
from wordparam import word2vector
from vggparam import vggparamater
from vggnet import VGGNet
import chainer
import chainer.functions as F
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ####単語とimg pathから1200次元ベクトルをつくる#####


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


# ####入力パラメータ#####
parser = argparse.ArgumentParser(
    description='Real-time style transfer image generator')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--dir', '-d', type=str, required=True,
                    help='output dir path')
args = parser.parse_args()

xp = np if args.gpu < 0 else cuda.cupy


# ########パラメータセット###########

batch = 0
batchsize = 10
device = args.gpu
n_epoch = 3
# a = wordQueryNet()

# 保存先をチェックする
if os.path.exists('models/' + args.dir):
    print('selected dir exists!')
else:
    print('selected dir does not exits! make dir.')
    os.mkdir('models/' + args.dir)

# 各種必要なパラメータを読み込み

# VGGmodelを読み込む
print('loading VGG model...')
vgg = VGGNet()
serializers.load_hdf5('VGG.model', vgg)
print('loaded VGG!')


# ############学習の高速化のためにパラメータを事前に整理しておく###################

print('preprocessing vgg data')
vgg_img_param = []
words = []

start = time.time()
for filename in filenames:
    vgg_img_param.append(vggparamater(filename, args.gpu, vgg)[0])

    if(filename.split('/')[7] == 'fabric'):
        word = '布'
    if(filename.split('/')[7] == 'foliage'):
        word = '植物'
    if(filename.split('/')[7] == 'glass'):
        word = 'ガラス'
    if(filename.split('/')[7] == 'leather'):
        word = '革'
    if(filename.split('/')[7] == 'metal'):
        word = '金属'
    if(filename.split('/')[7] == 'paper'):
        word = '紙'
    if(filename.split('/')[7] == 'plastic'):
        word = 'プラスチック'
    if(filename.split('/')[7] == 'stone'):
        word = '石'
    if(filename.split('/')[7] == 'water'):
        word = '水'
    if(filename.split('/')[7] == 'wood'):
        word = '木'

    words.append(word)
calctime = time.time() - start
print('VGG time: ' + str(calctime), '[sec]')
print('vgg end')

#############################################################################
print('training start')
dataset = []
for i in range(len(filenames)):
    dataset.append([words[i], target_img_param[i], vgg_img_param[i]])

# dataset = [words, target_img_param, vgg_img_param]
tinynet = wordQueryNet()

Optimizer = optimizers.MomentumSGD(lr=0.1, momentum=0.9)
Optimizer.setup(tinynet)
if device >= 0:
    cuda.get_device(device).use()
    tinynet.to_gpu()

for epoch in range(n_epoch):
    start = time.time()
    Optimizer.lr *= 0.1
    batch = 0
    print('epoch:' + str(epoch) + ' learning rate: ' + str(Optimizer.lr))

    while(batch + batchsize) <= len(dataset):
        Lsum = Variable(xp.zeros((), dtype=np.float32))
        for data in dataset[batch:batch + batchsize]:
            tinynet.zerograds()
            styleparam = concatData(data[0], data[2])  # styleparm: (1,500)
            styleparam_g = cuda.to_gpu(styleparam)
            style_vector = tinynet(styleparam_g)

            data[1] = chainer.cuda.to_gpu(data[1])
            loss = F.mean_squared_error(style_vector, data[1])
            data[1] = chainer.cuda.to_cpu(data[1])
            Lsum += loss
        Lsum.backward()
        Optimizer.update()
        batch += batchsize
        if(batch % 100 == 0):
            print('batch loss: ' + str(Lsum.data / batchsize))
    loss_mean = Lsum.data / batchsize
    print('training loss in this epoch:' + str('%1.10f' % loss_mean))
    calctime = time.time() - start
    print('learning time in thins epoch: ' + str(calctime) + '[sec]')
    serializers.save_npz('models/{}/epoch_{}.model'.format(args.dir, epoch), tinynet)
serializers.save_npz('models/{}/final.model'.format(args.dir), tinynet)
