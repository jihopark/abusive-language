#!/usr/bin/env python
"""Functions related to handling data"""

import numpy as np
from sklearn.model_selection import train_test_split

def train_test_split_in_chunk(x, y, test_size, n=5):
    split_index = []
    split_size = int(len(x)/n)
    k = 0
    for i in range(n-1):
        k += split_size
        split_index.append(k)
    print("split index")
    print(split_index)

    x_chunks = np.array_split(x, split_index)
    y_chunks = np.array_split(y, split_index)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(n):
        _y = y_chunks[i]
        _x = x_chunks[i]
        print("divided into chunk shape _x: %s, _y: %s" % (str(_x.shape), str(_y.shape)))
        _x_train, _x_test, _y_train, _y_test = train_test_split(_x, _y,
                test_size=test_size)
        x_train.append(_x_train)
        y_train.append(_y_train)
        x_test.append(_x_test)
        y_test.append(_y_test)
    print("concatenating chunks")
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    print("merged and splitted to shape x_train:%s, y_train:%s, x_test:%s, y_test:%s" %
            (str(x_train.shape), str(y_train.shape), str(x_test.shape), str(y_test.shape)))
    return x_train, x_test, y_train, y_test

def split_dataset_binary(x=None, x_neg=None, x_pos=None, split=[0.85, 0.15]):
    assert (split[0] + split[1]) == 1
    assert (x is not None or (x_neg is not None and x_pos is not None))
    if x_neg is None and x_pos is None:
        print("Need to split dataset by label")
        for i, row in enumerate(x):
            if row[-1] == 1:
                x = x[:, :-1] # remove labels
                x_pos, x_neg = np.split(x, [i])
                break
        print("Splitted to two arrays of shape %s and %s"
                % (str(x_neg.shape), str(x_pos.shape)))

    x_neg = np.array(x_neg)
    x_pos = np.array(x_pos)

    y_neg = np.zeros((len(x_neg),1))
    y_pos = np.ones((len(x_pos),1))

    # create training set
    if len(x_neg) > 5000:
        x_neg_train, x_neg_test, y_neg_train, y_neg_test = train_test_split_in_chunk(x_neg, y_neg, test_size=split[1])
    else:
        x_neg_train, x_neg_test, y_neg_train, y_neg_test = train_test_split(x_neg, y_neg, test_size=split[1])
    if len(x_pos) > 5000:
        x_pos_train, x_pos_test, y_pos_train, y_pos_test = train_test_split_in_chunk(x_pos, y_pos, test_size=split[1])
    else:
        x_pos_train, x_pos_test, y_pos_train, y_pos_test = train_test_split(x_pos, y_pos, test_size=split[1])

    x_train = np.concatenate([x_neg_train, x_pos_train])
    y_train = np.concatenate([y_neg_train, y_pos_train])

    x_test = np.concatenate([x_neg_test, x_pos_test])
    y_test = np.concatenate([y_neg_test, y_pos_test])

    print("Training set size:%s (neg:%s/pos:%s)\n" % (len(x_train), len(x_neg_train), len(x_pos_train)))
    print("sample neg - " + x_train[0])
    print("sample pos - " + x_train[-1])

    print("Test set size:%s (neg:%s/pos:%s)\n" % (len(x_test), len(x_neg_test), len(x_pos_test)))
    print("sample neg - " + x_test[0])
    print("sample pos - " + x_test[-1])


    return x_train, y_train, x_test, y_test

#TODO: create multi-class dataset split function
