import codecs
import pdb
import pickle
import sys

from tqdm import tqdm

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import (strip_multiple_whitespaces,
                                          strip_punctuation)
from gensim.test.utils import common_texts, get_tmpfile

sys.stdout = codecs.getwriter("utf-8")(sys.stdout)


def main():
    with open('/tmp/enwiki_reduced.txt', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.rstrip() for line in lines]
    parsed = []
    for line in tqdm(lines):
        words = []
        sentences = [sentence for sentence in line.split('.') if len(sentence) > 0]
        for sentence in tqdm(sentences):
            # replace punctuations with whitespace
            sentence = strip_punctuation(sentence)
            # remove multi white space
            sentence = strip_multiple_whitespaces(sentence)
            words = [word for word in sentence.split(' ') if len(word) > 0]
            parsed.append(words)
    with open('parsed.pickle', 'wb') as f:
        pickle.dump(parsed, f,protocol=-1)


if __name__ == '__main__':
    main()
