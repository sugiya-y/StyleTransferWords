from gensim.parsing.preprocessing import strip_punctuation, remove_stopwords, strip_multiple_whitespaces
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# read english wikipedia data
f = open('/tmp/enwiki.txt', encoding='utf-8')
lines = f.readlines()
f.close()
print(type(lines))
reduced_list = []
print('file read. start reducing')
for line in lines:
    # remove comma and period and so on
    reduced = strip_punctuation(line)
    # remove stop words
    reduced = remove_stopwords(reduced)
    # remove repeating white space
    reduced = strip_multiple_whitespaces(reduced)
    # save line
    reduced_list.append(reduced)
print('finish reducing. start save')
f = open('enwiki_reduced.txt', 'w', encoding='utf-8')
f.writelines(reduced_list)
f.close()