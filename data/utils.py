#!/usr/bin/env python
"""Functions related to handling data"""

import numpy as np
from sklearn.model_selection import train_test_split
from random import sample

def batch_gen(x, y, batch_size):
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size].T, y[i : (i+1)*batch_size].T

def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield [x[i] for i in sample_idx], [y[i] for i in sample_idx]

# batch generator that gives out balanced batch for each class
def balanced_batch_gen(x, y, batch_size, balance=[0.5, 0.5]):
    classes = np.unique(y)

    # for now only works for binary classes with even number of batch_size
    assert len(classes) == 2 and batch_size % 2 == 0

    idx_1 = np.where( y == classes[0])[0]
    idx_2 = np.where( y == classes[1])[0]
    x_1 = [x[i] for i in idx_1]
    x_2 = [x[i] for i in idx_2]
    y_1 = [y[i] for i in idx_1]
    y_2 = [y[i] for i in idx_2]

    print("Generating batch of %s with distribution of %.2f %.2f" %
            (batch_size, balance[0], balance[1]))

    while True:
        sample_idx_1 = sample(list(np.arange(len(x_1))),
                int(batch_size*balance[0]))
        sample_idx_2 = sample(list(np.arange(len(x_2))),
                int(batch_size*balance[1]))
        yield (np.concatenate(
                ([x_1[i] for i in sample_idx_1],
                 [x_2[i] for i in sample_idx_2])),
               np.concatenate(
                ([y_1[i] for i in sample_idx_1],
                 [y_2[i] for i in sample_idx_2])))


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

def split_dataset_binary(x=None, x_neg=None, x_pos=None, split=[0.7, 0.15, 0.15]):
    assert (split[0] + split[1] + split[2]) == 1
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

    y_neg = np.zeros(len(x_neg))
    y_pos = np.ones(len(x_pos))

    # create training set
    if len(x_neg) > 5000:
        x_neg_train, x_neg_test, y_neg_train, y_neg_test = train_test_split_in_chunk(x_neg, y_neg, test_size=split[1]+split[2])
    else:
        x_neg_train, x_neg_test, y_neg_train, y_neg_test = train_test_split(x_neg, y_neg, test_size=split[1]+split[2])
    if len(x_pos) > 5000:
        x_pos_train, x_pos_test, y_pos_train, y_pos_test = train_test_split_in_chunk(x_pos, y_pos, test_size=split[1]+split[2])
    else:
        x_pos_train, x_pos_test, y_pos_train, y_pos_test = train_test_split(x_pos, y_pos, test_size=split[1]+split[2])

    x_train = np.concatenate([x_neg_train, x_pos_train])
    y_train = np.concatenate([y_neg_train, y_pos_train])

    x_pos_valid, x_pos_test, y_pos_valid, y_pos_test = train_test_split(x_pos_test, y_pos_test,
                                                                        test_size=split[2]/(split[1]+split[2]))
    x_neg_valid, x_neg_test, y_neg_valid, y_neg_test = train_test_split(x_neg_test, y_neg_test,
                                                                        test_size=split[2]/(split[1]+split[2]))

    x_valid = np.concatenate([x_neg_valid, x_pos_valid])
    y_valid = np.concatenate([y_neg_valid, y_pos_valid])

    x_test = np.concatenate([x_neg_test, x_pos_test])
    y_test = np.concatenate([y_neg_test, y_pos_test])

    print("Training set size:%s (neg:%s/pos:%s)\n" % (len(x_train), len(x_neg_train), len(x_pos_train)))
    print("sample neg - " + x_train[0])
    print("sample pos - " + x_train[-1])

    print("Valid set size:%s (neg:%s/pos:%s)\n" % (len(x_valid), len(x_neg_valid), len(x_pos_valid)))
    print("sample neg - " + x_valid[0])
    print("sample pos - " + x_valid[-1])


    print("Test set size:%s (neg:%s/pos:%s)\n" % (len(x_test), len(x_neg_test), len(x_pos_test)))
    print("sample neg - " + x_test[0])
    print("sample pos - " + x_test[-1])


    return x_train, y_train, x_valid, y_valid, x_test, y_test

#TODO: create multi-class dataset split function
