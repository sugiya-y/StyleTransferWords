import random
import numpy as np
import os
import _pickle as cPickle
import re
import time

with open('yahoo100m_txt_lines.pickle', mode='rb') as f:
    print('text loading from pickle...')
    lines = cPickle.load(f)

with open('yahoo100m_adv_exists.pickle', mode='rb') as f:
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
            dataset.append([adv, imgid])
            count += 1
            if count >= 1000:
                break
    print('serch time for [', adv , '] is', str(time.time() - start))

