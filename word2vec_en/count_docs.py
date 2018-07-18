import nltk

with open("enwiki.txt", 'r', encoding='utf-8') as f:
    text = f.read() 

words_tok = nltk.word_tokenize(text)
words_tag = nltk.pos_tag(words_tok)

adv_list = []
for word_tag in words_tag:
    if word_tag[1] == 'JJ' or word_tag[1] == 'JJR' or word_tag[1] == 'JJS':
        # print(word_tag[0])
        adv_list.append(word_tag[0] + ' ')

print('text search finished. saving...')
with open('enwiki_adv.txt', 'w', encoding='utf-8') as f:
    f.writelines(adv_list)

print('saved text sorting and overwriting...')
with open('enwiki_adv.txt', 'r', encoding='utf-8') as f:
    advs = f.read()
advs_tok = nltk.word_tokenize(advs)
freqadvs = nltk.FreqDist(advs_tok)

sortedadvs = sorted(freqadvs.items(), key=lambda x: -x[1])
sortedlist = []
for sortedadv in sortedadvs:
    sortedlist.append(sortedadv[0] + ' ')
with open('enwiki_adv.txt', 'w', encoding='utf-8') as f:
    f.writelines(sortedlist)

