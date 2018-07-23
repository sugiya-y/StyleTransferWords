import nltk
from nltk.tag.perceptron import PerceptronTagger
import os
import _pickle as cPickle

if os.path.exists('text8_txt.pickle'):
    with open('text8_txt.pickle', mode='rb') as f:
        print('text loading from pickle...')
        text = cPickle.load(f)
else:
    with open("/tmp/text8", 'r', encoding='utf-8') as f:
        print('text loading from row data...')
        text = f.read() 
    with open('text8_txt.pickle', mode='wb') as f:
        print('text to pickle...')
        cPickle.dump(text, f, protocol=-1)
print('text loaded.')

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
print('text tokenized and tagged.')

print('searching adverbs...')
adv_list = []
for word_tag in words_tag:
    # if word_tag[1] == 'JJ' or word_tag[1] == 'JJR' or word_tag[1] == 'JJS':
    if word_tag[1] == 'JJ':
        # print(word_tag[0])
        adv_list.append(word_tag[0] + ' ')

print('text search finished. saving...')
with open('text8_adv.txt', 'w', encoding='utf-8') as f:
    f.writelines(adv_list)

print('saved text sorting and overwriting...')
with open('text8_adv.txt', 'r', encoding='utf-8') as f:
    advs = f.read()
advs_tok = nltk.word_tokenize(advs)
freqadvs = nltk.FreqDist(advs_tok)

sortedadvs = sorted(freqadvs.items(), key=lambda x: -x[1])
sortedlist = []
sortedlist_with_num = []
for sortedadv in sortedadvs:
    sortedlist.append(sortedadv[0] + ' ')
    sortedlist_with_num.append(sortedadv[0] + ': ' + str(sortedadv[1]) + '\n')
with open('text8_adv.txt', 'w', encoding='utf-8') as f:
    f.writelines(sortedlist)
with open('text8_adv_num.txt', 'w', encoding='utf-8') as f:
    f.writelines(sortedlist_with_num)
