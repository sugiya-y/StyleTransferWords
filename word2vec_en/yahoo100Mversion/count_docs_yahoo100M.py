import os
import _pickle as cPickle
import re
import nltk
from nltk.tag.perceptron import PerceptronTagger

if os.path.exists('yahoo100m_txt.pickle'):
    with open('yahoo100m_txt.pickle', mode='rb') as f:
        print('text loading from pickle...')
        text = cPickle.load(f)
else:
    with open("/tmp/yahoo100m", 'r', encoding='utf-8') as f:
        print('text loading from row data...')
        text = f.read() 
    with open('yahoo100m_txt.pickle', mode='wb') as f:
        print('text to pickle...')
        cPickle.dump(text, f, protocol=-1)
print('text loaded.')

words_list = " "
sentences = text.split('\n')
for sentence in sentences:
    splitted = sentence.split('\t')
    # print(splitted[-1].encode('utf-8').decode())
    if len(splitted) == 11:
        predwords = re.sub('[0-9:.]', '', splitted[10])
        labelwords = re.sub('[0-9:+-]', ' ', splitted[9])
        words = labelwords + ' ' + predwords + ' '
        words_list += words

tagger = PerceptronTagger()
tagset = None
words_tag = []
# counter = 0
# sentences = text.split(".")
# length = len(sentences)
# for sentence in sentences:
#     counter += 1
#     print(counter, '/', length)
#     words_tok = nltk.word_tokenize(sentence)
#     words_tag_sent = nltk.tag._pos_tag(words_tok, tagset, tagger)
#     words_tag.extend(words_tag_sent)

words_tok = nltk.word_tokenize(words_list)
print('text tokenized..')
words_tag = nltk.tag._pos_tag(words_tok, tagset, tagger)

print('text tokenized and tagged.')

print('searching adverbs...')
adv_list = []
for word_tag in words_tag:
    # if word_tag[1] == 'JJ' or word_tag[1] == 'JJR' or word_tag[1] == 'JJS':
    if word_tag[1] == 'JJ':
        # print(word_tag[0])
        adv_list.append(word_tag[0] + ' ')

print('text search finished. saving...')
with open('yahoo100m_adv.txt', 'w', encoding='utf-8') as f:
    f.writelines(adv_list)

print('saved text sorting and overwriting...')
with open('yahoo100m_adv.txt', 'r', encoding='utf-8') as f:
    advs = f.read()
advs_tok = nltk.word_tokenize(advs)
freqadvs = nltk.FreqDist(advs_tok)

sortedadvs = sorted(freqadvs.items(), key=lambda x: -x[1])
sortedlist = []
sortedlist_with_num = []
for sortedadv in sortedadvs:
    sortedlist.append(sortedadv[0] + '\n')
    sortedlist_with_num.append(sortedadv[0] + ': ' + str(sortedadv[1]) + '\n')
with open('yahoo100m_adv.txt', 'w', encoding='utf-8') as f:
    f.writelines(sortedlist)
with open('yahoo100m_adv_num.txt', 'w', encoding='utf-8') as f:
    f.writelines(sortedlist_with_num)