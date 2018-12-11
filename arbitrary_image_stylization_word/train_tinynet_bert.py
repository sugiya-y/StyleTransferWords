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
sys.path.append('../word2vec_en/leatherversion/')
from dataselector import dataselect
from seq2vec import seq2vec

# chainer関連をimportする前にデータの前処理をしておかないとなんかエラー吐くのでここは譲れない

start = time.time()
word_id_sets = dataselect()
filenames = []
words = []

for word_id_set in word_id_sets:
    try:
        # assert word_id_set[1].shape[2] == 3
        words.append(word_id_set[0])
        filenames.append(word_id_set[1])
    except:
        continue

# print(filenames)
# print('content image loaded!')
# print('preprocessing target style data')
# filenames = filenames[0:12]  # for testing
if os.path.isfile('features/target_leather.pickle'):
    # print('inception checkpoint pickle is exist! loading pickle')
    with open('features/target_leather.pickle', mode='rb') as f:
        target_img_param = pickle.load(f)
else:
    target_img_param = styleParam(filenames)
    with open('features/target_leather.pickle', mode='wb') as f:
        pickle.dump(target_img_param, f)
calctime = time.time() - start
# print('inception v3 time: ' + str(calctime) + '[sec]')

import argparse
from PIL import ImageFile
from chainer import cuda, Variable, optimizers, serializers
from gensim.models import word2vec
from vggparam import vggparamater
from vggnet import VGGNet
from tinygan_bert import Generator, Discriminator
import chainer
import chainer.functions as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
import codecs

sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


# ####単語とimg pathから1200次元ベクトルをつくる#####

# model = word2vec.Word2Vec.load("../word2vec_en/enwiki.model")

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
    vec = seq2vec(word)
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
batchsize = 1000
device = args.gpu
n_epoch = 81

# # 保存先をチェックする
# if os.path.exists('models/' + args.dir):
#     # print('selected dir exists!')
#     # print(batchsize)
# else:
#     # print('selected dir does not exits! make dir.')
#     os.mkdir('models/' + args.dir)
if os.path.exists('models/' + args.dir) is False:
    os.mkdir('models/' + args.dir)

# 各種必要なパラメータを読み込み

# VGGmodelを読み込む
if args.usevgg == 1:
    # print(str('loading VGG model...'))
    vgg = VGGNet()
    serializers.load_hdf5('/tmp/VGG.model', vgg)
    # print('loaded VGG!')


# ############学習の高速化のためにパラメータを事前に整理しておく###################
if args.usevgg == 1:
    # print('preprocessing vgg data')
    vgg_img_param = []

    start = time.time()
    if os.path.isfile('features/vggparam_leather.pickle'):
        # print('vgg checkpoint pickle is exist! loading pickle')
        with open('features/vggparam_leather.pickle', mode='rb') as f:
            vgg_img_param = pickle.load(f)
    else:
        for filename in filenames:
            vgg_img_param.append(vggparamater(filename, args.gpu, vgg)[0])
        with open('features/vggparam_leather.pickle', mode='wb') as f:
            pickle.dump(vgg_img_param, f)
    calctime = time.time() - start
    # print('VGG time: ' + str(calctime), '[sec]')
    # print('vgg end')

#############################################################################
# print('training start')
dataset = []

style = np.load('style.npy')
if os.path.isfile('styleg_bert.npy'):
    # print('the file is exists! load...')
    styleg = np.load('styleg_bert.npy')
else:
    # print('features file not exists! making...')
    if args.usevgg == 1:
        styleg = np.array(concatData(words[0], vgg_img_param[0]))
    else:
        styleg = np.array(w2v(words[0]))
    # import pdb
    # pdb.Pdb(stdout=sys.__stdout__).set_trace()
    # style = np.reshape(np.array(target_img_param[0]), (1, 100))

    if args.usevgg == 1:
        for i in range(1, len(filenames)):
            styleg = np.vstack((styleg, concatData(words[i], vgg_img_param[i])))
            # style = np.vstack((style, np.reshape(target_img_param[i], (1, 100))))
    else:
        for i in range(1, len(filenames)):
            styleg = np.vstack((styleg, w2v(words[i])))
            # style = np.vstack((style, np.reshape(target_img_param[i], (1, 100))))
    np.save('styleg_bert', styleg)
    # np.save('style_bert', style)

# print(styleg.shape, style.shape)

# dataset.append([words[i], target_img_param[i], vgg_img_param[i]])
# dataset = chainer.cuda.to_gpu(dataset)
# dataset = [words, target_img_param, vgg_img_param]
if args.usevgg == 1:
    gen = Generator()
    dis = Discriminator()
else:
    gen = Generator()
    dis = Discriminator()

# print('a')
# Optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)
Optimizer_gen = optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
Optimizer_gen.setup(gen)
Optimizer_dis = optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
Optimizer_dis.setup(dis)
if device >= 0:
    cuda.get_device(device).use()
    gen.to_gpu()
    dis.to_gpu()

debug = False

for epoch in range(n_epoch):
    start = time.time()
    # Optimizer.lr *= 0.1
    batch = 0
    ct = 0
    # print('epoch:' + str(epoch) + ' learning rate: ' + str(Optimizer.lr))
    # print('epoch:' + str(epoch))
    Lsum_gen = Variable(xp.zeros((), dtype=np.float32))
    Lsum_dis = Variable(xp.zeros((), dtype=np.float32))
    while(batch + batchsize) <= style.shape[0]:
        ct = ct + 1
        gen.zerograds()
        dis.zerograds()
        styleparam_g = Variable(chainer.cuda.to_gpu(
            styleg[batch:batch + batchsize]))
        styleparam = Variable(chainer.cuda.to_gpu(
            style[batch:batch + batchsize]))
        style_vector = gen(styleparam_g)
        dis_out1 = dis(style_vector) # にせもの
        dis_out2 = dis(styleparam) # ほんもの
        # if debug:
            # print('params:')
            # print(style_vector.data[2][0:10])
            # print(styleparam.data[2][0:10])
        # print(style_vector.data.shape,styleparam.data.shape)

        #    da = chainer.cuda.to_gpu(data[1])
        loss_gen = F.softmax_cross_entropy(dis_out1, Variable(xp.zeros(batchsize, dtype=np.int32))) # 生成したパラメータがどうなのかadversarial loss, 0本物に近づけたい
        loss_dis = F.softmax_cross_entropy(dis_out1, Variable(xp.ones(batchsize, dtype=np.int32))) # 1(偽物)に近づけたい
        loss_dis += F.softmax_cross_entropy(dis_out2, Variable(xp.zeros(batchsize, dtype=np.int32))) # 本物をdiscriminatorに入力. 0(本物)に近づけたい
        # data[1] = chainer.cuda.to_cpu(data[1])
        Lsum_gen += loss_gen
        Lsum_dis += loss_dis
        if batch < style.shape[0] * 0.9:
            loss_gen.backward()
            Optimizer_gen.update()
            loss_dis.backward()
            Optimizer_dis.update()
        # else:
            # print("val loss: gen:%s  dis:%s" % (loss_gen.data, loss_dis.data))
        batch += batchsize
        # if(batch % 1000 == 0):
        #    print('batch loss: ' + str(loss.data))
    loss_gen_mean = Lsum_gen.data / ct
    loss_dis_mean = Lsum_dis.data / ct
    # print('training loss in this epoch [Generator]:' + str('%1.10f' % loss_gen_mean))
    # print('training loss in this epoch [Discriminator]:' + str('%1.10f' % loss_dis_mean))
    # print('epoch weight' + str(tinynet.W))
    calctime = time.time() - start
    # print('learning time in thins epoch: ' + str(calctime) + '[sec]')
    if epoch % 5 == 0:
        serializers.save_npz('models/{}/epoch_{}.model'.format(args.dir, epoch), gen)
serializers.save_npz('models/{}/final.model'.format(args.dir), gen)
