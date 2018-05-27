#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy


def word2vecter(in_word):

    with open('word2vec.model', 'r') as f:
        ss = f.readline().split()
        n_vocab, n_units = int(ss[0]), int(ss[1])
        word2index = {}
        index2word = {}
        w = numpy.empty((n_vocab, n_units), dtype=numpy.float32)
        for i, line in enumerate(f):
            ss = line.split()
            assert len(ss) == n_units + 1
            word = ss[0]
            word2index[word] = i
            index2word[i] = word
            w[i] = numpy.array([float(s) for s in ss[1:]], dtype=numpy.float32)

    s = numpy.sqrt((w * w).sum(1))
    w /= s.reshape((s.shape[0], 1))  # normalize

    try:
        while True:
            if in_word not in word2index:
                print('"{0}" is not found'.format(in_word))
                continue
            v1 = w[word2index[in_word]]

            return v1

    except EOFError:
        pass
        return 0
