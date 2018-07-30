#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding:utf-8

# まずは画像のパラメータを拾得する
from generate_style_parameter import styleParam
import numpy as np
import time
import os
import pickle
import sys
sys.path.append('../word2vec_en/yahoo100Mversion/')
from dataselector import dataselect

# chainer関連をimportする前にデータの前処理をしておかないとなんかエラー吐くのでここは譲れない

start = time.time()
word_id_sets = dataselect()
filenames = []
words = []

for word_id_set in word_id_sets:
    words.append(word_id_set[0])
    filenames.append(word_id_set[1])

# print(filenames)
print('content image loaded!')
print('preprocessing target style data')
# filenames = filenames[0:12]  # for testing
if os.path.isfile('features/target_yahoo.pickle'):
    print('inception checkpoint pickle is exist! loading pickle')
    with open('features/target_yahoo.pickle', mode='rb') as f:
        target_img_param = pickle.load(f)
else:
    target_img_param = styleParam(filenames)
    with open('features/target_yahoo.pickle', mode='wb') as f:
        pickle.dump(target_img_param, f)
calctime = time.time() - start
print('inception v3 time: ' + str(calctime) + '[sec]')

import argparse
from PIL import ImageFile
from chainer import cuda, Variable, optimizers, serializers
from tinynet import wordQueryNet
from tinynet_novgg import wordQueryNetNoVGG
from gensim.models import word2vec
from vggparam import vggparamater
from vggnet import VGGNet
import chainer
import chainer.functions as F
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ####単語とimg pathから1200次元ベクトルをつくる#####

model = word2vec.Word2Vec.load("../word2vec_en/enwiki.model")

def concatData(word, vgg_img_param):
    vec = model.wv[word]

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
    vec = model.wv[word]
    vec = vec / np.linalg.norm(vec)

    return vec


# ####入力パラメータ#####
parser = argparse.ArgumentParser(
    description='Real-time style transfer image generator')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--dir', '-d', type=str, required=True,
                    help='output dir path')
parser.add_argument('--usevgg', '-u', default=1, type=int,
                    help='use or dont use vgg: 0 or 1')
args = parser.parse_args()

xp = np if args.gpu < 0 else cuda.cupy


# ########パラメータセット###########

batch = 0
batchsize = 50
device = args.gpu
n_epoch = 50
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
serializers.load_hdf5('/tmp/VGG.model', vgg)
print('loaded VGG!')


# ############学習の高速化のためにパラメータを事前に整理しておく###################

print('preprocessing vgg data')
vgg_img_param = []

start = time.time()
if os.path.isfile('features/vggparam_yahoo.pickle'):
    print('vgg checkpoint pickle is exist! loading pickle')
    with open('features/vggparam_yahoo.pickle', mode='rb') as f:
        vgg_img_param = pickle.load(f)
else:
    for filename in filenames:
        vgg_img_param.append(vggparamater(filename, args.gpu, vgg)[0])
    with open('features/vggparam_yahoo.pickle', mode='wb') as f:
        pickle.dump(vgg_img_param, f)
calctime = time.time() - start
print('VGG time: ' + str(calctime), '[sec]')
print('vgg end')

#############################################################################
print('training start')
dataset = []

if args.usevgg == 1:
    styleg = np.array(concatData(words[0], vgg_img_param[0]))
else:
    styleg = np.array(w2v(words[0]))
style = np.reshape(np.array(target_img_param[0]), (1, 100))

if args.usevgg == 1:
    for i in range(1, len(filenames)):
        styleg = np.vstack((styleg, concatData(words[i], vgg_img_param[i])))
        style = np.vstack((style, np.reshape(target_img_param[i], (1, 100))))
else:
    for i in range(1, len(filenames)):
        styleg = np.vstack((styleg, w2v(words[i])))
        style = np.vstack((style, np.reshape(target_img_param[i], (1, 100))))

print(styleg.shape, style.shape)

# dataset.append([words[i], target_img_param[i], vgg_img_param[i]])
# dataset = chainer.cuda.to_gpu(dataset)
# dataset = [words, target_img_param, vgg_img_param]
if args.usevgg == 1:
    tinynet = wordQueryNet()
else:
    tinynet = wordQueryNetNoVGG()

# print('a')
# Optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)
Optimizer = optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
Optimizer.setup(tinynet)
if device >= 0:
    cuda.get_device(device).use()
    tinynet.to_gpu()

debug = False

for epoch in range(n_epoch):
    start = time.time()
    # Optimizer.lr *= 0.1
    batch = 0
    ct = 0
    # print('epoch:' + str(epoch) + ' learning rate: ' + str(Optimizer.lr))
    print('epoch:' + str(epoch))
    Lsum = Variable(xp.zeros((), dtype=np.float32))
    while(batch + batchsize) <= style.shape[0]:
        ct = ct + 1
        tinynet.zerograds()
        styleparam_g = Variable(chainer.cuda.to_gpu(
            styleg[batch:batch + batchsize]))
        styleparam = Variable(chainer.cuda.to_gpu(
            style[batch:batch + batchsize]))
        style_vector = tinynet(styleparam_g)
        if debug:
            print('params:')
            print(style_vector.data[2][0:10])
            print(styleparam.data[2][0:10])
        # print(style_vector.data.shape,styleparam.data.shape)

        #    da = chainer.cuda.to_gpu(data[1])
        loss = F.mean_squared_error(style_vector, styleparam)
        # data[1] = chainer.cuda.to_cpu(data[1])
        Lsum += loss
        if batch < 9500:
            loss.backward()
            Optimizer.update()
        else:
            print("val loss: %s" % (loss.data))
        batch += batchsize
        # if(batch % 1000 == 0):
        #    print('batch loss: ' + str(loss.data))
    loss_mean = Lsum.data / ct
    print('training loss in this epoch:' + str('%1.10f' % loss_mean))
    # print('epoch weight' + str(tinynet.W))
    calctime = time.time() - start
    print('learning time in thins epoch: ' + str(calctime) + '[sec]')
    # serializers.save_npz('models/{}/epoch_{}.model'.format(args.dir, epoch), tinynet)
serializers.save_npz('models/{}/final.model'.format(args.dir), tinynet)
