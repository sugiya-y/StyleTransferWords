#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding:utf-8

import random
import numpy as np
import os
try:
    import _pickle as cPickle
except ImportError:
    import cPickle
import re
import time


def dataselect():
    if os.path.exists('/home/yanai-lab/sugiya-y/space/research/privateWork/word2vec_en/yahoo100Mversion/searched_words.pickle'):
        with open('/home/yanai-lab/sugiya-y/space/research/privateWork/word2vec_en/yahoo100Mversion/searched_words.pickle', mode='rb') as f:
            print('searched words loading from pickle...')
            dataset = cPickle.load(f)
    else:
        print('There is no searched.pickle!')
        with open('/home/yanai-lab/sugiya-y/space/research/privateWork/word2vec_en/yahoo100Mversion/yahoo100m_txt_lines.pickle', mode='rb') as f:
            print('text loading from pickle...')
            lines = cPickle.load(f)

        with open('/home/yanai-lab/sugiya-y/space/research/privateWork/word2vec_en/yahoo100Mversion/yahoo100m_adv_exists.pickle', mode='rb') as f:
            print('advs loading from pickle...')
            advs = cPickle.load(f)
        advs = advs[:499] # 上位500単語のみを使用
        print('adverbs loaded.')

        dataset = []
        print('shuffling lines...')
        random.shuffle(lines)
        print('serching words...')
        for adv in advs:
            adv = adv.rstrip('\n')
            start = time.time()
            count = 0
            for line in lines:
                if line.find(adv) != -1:
                    imgid = line.split('\t')[0]
                    imgpath = os.path.join('/export/data/dataset/Yahoo100M/photo', imgid[:2], imgid + '.jpg')
                    if os.path.exists(imgpath):
                        dataset.append([adv, imgpath])
                        count += 1
                    if count >= 1000:
                        break
            # print('serch time for [', adv , '] is', str(time.time() - start))
        with open('/home/yanai-lab/sugiya-y/space/research/privateWork/word2vec_en/yahoo100Mversion/searched_words.pickle', mode='wb') as f:
             cPickle.dump(dataset, f, protocol=-1)
    return(dataset)
