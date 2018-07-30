
import os
import _pickle as cPickle
import re
import nltk
from nltk.tag.perceptron import PerceptronTagger
from gensim.models import word2vec

with open('yahoo100m_adv.txt') as f:
    print('loading adverbs...')
    advs = f.readlines()
print('adverbs loaded.')

print('loading word2vec...')
model = word2vec.Word2Vec.load("../enwiki.model")

print('reducing advs...')
selected = []
for adv in advs:
    adv = adv.rstrip('\n')
    try:
        print(adv, 'is exists!')
        vec = model.wv[adv]
        selected.append(adv)
    except KeyError:
        print(adv, 'is NONE!')

with open('yahoo100m_adv_exists.pickle', mode='wb') as f:
    cPickle.dump(selected, f, protocol=-1)

count = 0
lines = []
for selectedadv in selected:
    count += 1
    selectedadv = str(count) + ': ' + selectedadv + '\n'
    lines.append(selectedadv)

with open('yahoo100m_adv_exists.txt', 'w') as f:
    print('saving....')
    f.writelines(lines)