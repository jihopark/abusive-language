#!/usr/bin/env python
""" Helper functions for combining word-based & character-based features for
HybridCNN training"""

import numpy as np

from . import char
from . import word

# make into list of dictionary
# into tuple of (?, word_text_len), (?, char_text_len,char_vocab_size)
def extract_from_batch(batch):
    batchW = []
    batchC = []
    for row in batch:
         batchW.append(row["word"])
         batchC.append(row["char"])
    return np.array(batchW), np.array(batchC)

def load_data_from_file(name):
    # load data from word_outputs
    (x_word_train, y_word_train, x_word_valid, y_word_valid, x_word_test,
        y_word_test, initW, vocab) = word.load_data_from_file(name)
    word_text_len = x_word_train.shape[1]
    word_vocab_size = len(vocab.vocabulary_)

    # load data from char_outputs
    x_char_train, y_char_train, x_char_valid, y_char_valid, x_char_test, y_char_test = char.load_data_from_file(name)
    char_text_len = x_char_train.shape[1]
    char_vocab_size = x_char_train.shape[2]

    # check whether data set has same lengths
    assert y_word_train.shape == y_char_train.shape
    assert y_word_valid.shape == y_char_valid.shape
    assert y_word_test.shape == y_char_test.shape

    # check whether all labels are in same order
    for y, y_ in [(y_word_train, y_char_train),
                  (y_word_valid, y_char_valid),
                  (y_word_test, y_char_test)]:
        for i in range(len(y)):
            assert y[i] == y_[i]
    print("\ndataset passed the assertion test")

    x_train = []
    x_valid = []
    x_test = []
    X = [(x_word_train, x_char_train, x_train),
        (x_word_valid, x_char_valid, x_valid),
        (x_word_test, x_char_test, x_test)]

    for w, c, h in X: 
        for i in range(len(w)):
            h.append({"word": w[i],
                      "char": c[i]})
    return (x_train, y_word_train,
            x_valid, y_word_valid,
            x_test, y_word_test,
            initW, vocab)

