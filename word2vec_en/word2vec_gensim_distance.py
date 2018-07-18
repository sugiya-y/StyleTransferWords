#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gensim.models import word2vec
import numpy as np
import codecs
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

model = word2vec.Word2Vec.load("./enwiki.model")
# cos類似度を計算するためのmost_similar
results = model.wv.most_similar(positive=['apple'])
# results = model.wv.get_vector('apple')
# f = codecs.open('./utftest.txt', 'w', 'utf-8')
for result in results:
    print(result[0], result[1])
    # f.write(result[0])
# f.close()

# print(np.shape(results))