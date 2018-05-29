#!/usr/bin/env python
# -*- coding: utf-8 -*-

# まずは画像のパラメータを拾得する
from generate_style_parameter import styleParam
print('preprocessing target style data')
tar_img_param = styleParam('images/style_images_own/*.jpg')

import numpy as np
import os
import argparse
from PIL import Image
import sys
import cv2 as cv
from PIL import ImageFile
from chainer import cuda, Variable, optimizers, serializers
# from net import *
# from small import *
# from netXX import *
# import marshal
from wordparam import word2vector
from vggparam import vggparamater
from vggnet import VGGNet
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ####単語とimg pathから1200次元ベクトルをつくる#####


def concatData(word, cont_img_param, g, vgg):
    # 時間がかかるのでデータがあれば読み込む
    if os.path.isfile('wordparam/word2vecter' + word + '.npy'):
        vec = np.load('wordparam/word2vecter' + word + '.npy')
        # print 'word2vecter loaded!!'
    else:
        vec = word2vector(word)
        # 時間がかかるのでデータを保存する
        np.save('wordparam/word2vecter' + word + '.npy', vec)
        # print 'word2vecter saved!!'

    # print 'VGG calculating...'
    # imgquery = vggparamater(cont_img,g,vgg)
    # print 'VGG complete!!'

    param = np.zeros((450, 1))
    vec = vec * 40
    # print np.shape(vec) , np.shape(cont_img_param)
    concated = np.concatenate((vec, cont_img_param[0]))
    param = np.reshape(concated, (450, 1))

    return np.transpose(param)


# ####入力パラメータ#####
parser = argparse.ArgumentParser(
    description='Real-time style transfer image generator')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--output', '-o', default='out', type=str,
                    help='output model file path without extension')
parser.add_argument('--dir', '-d', type=str, required=True,
                    help='output dir path')
parser.add_argument('--model', '-m', default='photo', type=str,
                    help='choose unseen model. "art" or "photo"')
args = parser.parse_args()

# if args.model == 'photo':
    # unseen_model_path = '../Unseen/unseen_style_transfer/models/Pretrained_photo/out_final.model'
# elif args.model == 'art':
    # unseen_model_path = '../Unseen/unseen_style_transfer/models/unseen.model'
# else:
    # print('enter model type with "photo" or "art" in -m args')
    # sys.exit()

# unseen_model = FastStyleNetSmall()
# serializers.load_npz(unseen_model_path, unseen_model)

# comm = chainermn.create_communicator()
# device = comm.intra_rank

# ########パラメータセット###########

batch = 0
batchsize = 10
device = args.gpu
savecount = 0
epoch = 0
# a = wordQueryNet()

# 保存先をチェックする
if os.path.exists('models/' + args.dir):
    print('selected dir exists!')
else:
    print('selected dir does not exits! make dir.')
    os.mkdir('models/' + args.dir)


# serializers.save_npz(
    # 'models/{}/{}_{}_{}.model'.format(args.dir, args.output, epoch, savecount), a)

# 各種必要なパラメータを読み込み

# VGGmodelを読み込む
print('loading VGG model...')
vgg = VGGNet()
serializers.load_hdf5('VGG.model', vgg)
print('loaded VGG!')

# wordsに形容詞を読み込む
# print('words loading...')
# f = open('images/labels.txt')
# words = f.readlines()
# f.close()
# for i in range(len(words)):
    # print words[i]
    # if (words[i] == '\n'):
        # del words[i]

# print('words loaded!')

# imgに画像pathをたくさん読み込む
print('content image loading...')
"""
con_img_paths = []
for j in range(len(words)):
    con_img_paths.append([])
    print('searching: dataset/art/' + str(j) + '/')
    for i in range(1000):  # ここを変えるとおかしくなる気がする
        if os.path.isdir('dataset/art/' + str(j)):
            if os.path.isfile('dataset/art/' + str(j) + '/' + str(i) + '.jpg'):
                con_img_paths[j].append(
                    'dataset/art/' + str(j) + '/' + str(i) + '.jpg')
                # print con_img_paths[j]
            if os.path.isfile('dataset/art/' + str(j) + '/' + str(i) + '.png'):
                con_img_paths[j].append(
                    'dataset/art/' + str(j) + '/' + str(i) + '.png')
"""
filenames = np.load('images/filenames.npy')
print('content image loaded!')
# print con_img_paths[0]

"""
if (len(words) != len(con_img_paths)):
    print('ワードリストか画像リストがおかしい！！！！！！！！！！')
    # print 'word: '
    # print words
    print(' shape: ')
    print(np.shape(words))
    # print 'con_img_paths: '
    # print con_img_paths
    print(' shape: ')
    print(np.shape(con_img_paths))
    sys.exit()
"""

# ############学習の高速化のためにパラメータを事前に整理しておく###################

print('preprocessing vgg data')
vgg_img_param = []

for filename in filenames:
    vgg_img_param.append(vggparamater(filename, args.gpu, vgg))
# vgg_img_param is (num_img, 1, 300)

print(np.shape(vgg_img_param))
# print cont_img_param[0]
#############################################################################
'''
dataset = [words, con_img_paths, cont_img_param]

train = []
count = 0
for i in range(len(dataset[0])):
    for j in range(len(dataset[1][i])):
        train.append([dataset[0][i], dataset[1][i][j], dataset[2][count], i])
        count += 0

np.random.shuffle(train)


# print train
# ###########計算が遅いからデータを減らします##############
# train = train[:][0:int(0.7*len(train))]

O = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
O.setup(a)
if device >= 0:
    chainer.cuda.get_device(device).use()
    a.to_gpu()

# #####学習ループ開始#####
n_epoch = 5
for epoch in range(n_epoch):
    O.lr *= 1 / 10.0
    batch = 0
    print('epoch:' + str(epoch) + ' learning rate: ' + str(O.lr))

    while (batch + batchsize) <= len(train):
        # print 'training in [' + str(batch) + ':' + str(batch+batchsize) +']'
        # print np.shape(train[batch:batch+batchsize])
        for train_batch in train[batch:batch + batchsize]:
            # print train_batch
            Lsum = Variable(xp.zeros((), dtype=np.float32))
            # print wordcount
            word = train_batch[0].rstrip('\n')
            cont_img = train_batch[1]
            cont_param = train_batch[2]
            # ######unseen style transferのパラメータを取得########
            """
            cimg = cv.imread(cont_img)

            if (cimg is None):
                #print 'image is None. data skipped'
                break
            """
            image = xp.asarray(Image.open(cont_img).convert('RGB').resize(
                (256, 256)), dtype=xp.float32).transpose(2, 0, 1)
            image = image.reshape((1,) + image.shape)

            x = Variable(image, volatile=False)

            param = concatData(word, cont_param, device, vgg)
            param_g = chainer.cuda.to_gpu(param)

            a.zerograds()
            b = a(param_g)

            for sty_img in dataset[1][train_batch[3]]:
                """
                if np.random.rand() > 0.99:
                    #print 'rand'
                    continue
                """
                style0 = xp.asarray(Image.open(sty_img).convert('RGB').resize(
                    (256, 256)), dtype=np.float32).transpose(2, 0, 1)
                style0 = style0.reshape((1,) + style0.shape)

                x2 = Variable(style0, volatile=False)

                y = unseen_model(x, x2, False)

                simg = cv.imread(sty_img)
                if ((cimg is None) or (simg is None)):
                    # print 'image is None. data skipped'
                    break

                loss = F.mean_squared_error(b, y)

                if (loss.data != loss.data):
                    print('loss is nan')
                    sys.exit()

                Lsum += loss

            Lsum.backward()

            O.update()

        loss_mean = Lsum.data / batchsize
        print('training loss:' + str('%1.10f' % loss_mean) + ' epoch:' + str(epoch) + ': ' + str('%3.6f' % (float(batch) / len(train) * 100)) + '%')
        batch += batchsize
        if batch % 400 == 0:
            savecount += 1
            serializers.save_npz(
                'models/{}/{}_{}_{}.model'.format(args.dir, args.output, epoch, savecount), a)

serializers.save_npz(
    'models/{}/{}_final.model'.format(args.dir, args.output), a)
# print y.dtype , y.shape
# print b.dtype , b.shape
print("end")
'''
