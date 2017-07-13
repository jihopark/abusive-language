#!/usr/bin/env python
""" Helper functions for character-based features"""

import os

import numpy as np
from . import preprocess

TWEET_MAX_LEN = 140
FEATURES = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’’’/\|_@#$%ˆ&*~‘+-=<>()[]{}")
N_DIM = len(FEATURES)

# turns 1hot row into char
def one_hot_to_char(row):
    for i, c in enumerate(row):
        if c == 1:
            return FEATURES[i]
    return ''

# turns 1hot 2d matrix into chars
def one_hot_to_chars(mat):
    return [one_hot_to_char(_row) for _row in mat]

def text_to_1hot_matrix(text, max_len=TWEET_MAX_LEN):
    tokens = list(filter(lambda x: x in FEATURES, list(text)))
    matrix = np.zeros((max_len, N_DIM))
    for i, t in enumerate(tokens):
        if i < max_len:
            row = np.zeros(N_DIM)
            try:
                j = FEATURES.index(t)
            except ValueError:
                j = -1
            if j >= 0:
                row[j] = 1
            matrix[i] = row
    return matrix

