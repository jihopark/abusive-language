#!/usr/bin/env python
"""Functions related to handling data"""

import numpy as np
from sklearn.model_selection import train_test_split
from random import sample
from . import char
from . import word

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
def balanced_batch_gen(x, y, batch_size, num_classes=2):
    classes = np.unique(y)

    assert len(classes) == num_classes and batch_size % num_classes == 0

    idx = []
    _x = []
    _y = []
    for i in range(num_classes):
        idx.append(np.where( y == classes[i])[0])
        _x.append([x[j] for j in idx[i]])
        _y.append([y[j] for j in idx[i]])

    balance = 1 / num_classes
    print("Generating balanced batch with n_classes=%s, balance=%.3f" %
            (num_classes, balance))
    while True:
        sample_idx = []
        for cl in range(num_classes):
            sample_idx.append(sample(list(np.arange(len(_x[cl]))),
                int(batch_size*balance)))

        batch_x = []
        batch_y = []
        for cl in range(num_classes):
            batch_x += [_x[cl][i] for i in sample_idx[cl]]
            batch_y += [_y[cl][i] for i in sample_idx[cl]]

        yield batch_x, batch_y

def print_errors(x, true, pred, model_name, dictionary=None):
    def print_sample(row):
        if model_name == "char_cnn":
            print(''.join(char.one_hot_to_chars(row)))
        else:
            print(" ".join(word.one_hot_to_words(row, dictionary)))

    print("\nError Analysis")
    errors = np.hstack((true, np.array(pred).reshape((len(pred), 1))))
    # get some true positives
    correct_idx = errors[:, 0] == errors[:, 1]
    correct_x = x[correct_idx]
    correct_pred = errors[correct_idx]
    tp_idx = correct_pred[:, 0] == 1
    tp = correct_x[tp_idx]

    print("\nTrue Positives")
    if len(tp) > 0:
        n_sample = len(tp) if len(tp) < 5 else 5
        tp_sample = tp[np.random.choice(len(tp), n_sample)]
        for row in tp_sample:
            print_sample(row)


    # leave only the wrong predictions
    error_idx = errors[:, 0] != errors[:, 1]
    errors = errors[error_idx]
    _x = x[error_idx]

    print("\nFalse Positives")
    fp = _x[errors[:, 1] == 1]
    if len(fp) > 0:
        n_sample = len(fp) if len(fp) < 5 else 5
        fp = fp[np.random.choice(len(fp), n_sample)]
        for row in fp:
            print_sample(row)

    print("\nFalse Negatives")
    fn = _x[errors[:, 1] == 0]
    if len(fn) > 0:
        n_sample = len(fn) if len(fn) < 5 else 5
        fn = fn[np.random.choice(len(fn), n_sample)]
        for row in fn:
            print_sample(row)

