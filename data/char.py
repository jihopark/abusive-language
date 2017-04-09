#!/usr/bin/env python
""" Helper functions for character-based features"""

import os

import numpy as np
from . import tokenizer
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

def text_to_1hot_matrix(text, max_len):
    tokens = tokenizer.to_chars(text)
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

def make_char_1hot_matrix(data, labels, split_name, file_path):
    print(data[0])
    print(data[-1])

    print("\n%s Texts to 1hot_matrix" % split_name)
    # transform text into 1hot matrix
    data = list(map(lambda x: text_to_1hot_matrix(x, TWEET_MAX_LEN), data))

    # save into file
    np.save(file_path + "/%s_data.npy" % split_name, data)
    print("saved to " + file_path + ("/%s_data.npy" % split_name))
    np.save(file_path + "/%s_labels.npy" % split_name,
            labels.reshape(len(labels), 1))
    print("saved to " + file_path + ("/%s_labels.npy" % split_name))



def preprocess_char_cnn(splitted_data, data_name):
    print("\n\nCreating Character 1-hot matrix for data=" + data_name)

    # check if already there
    file_path = os.path.dirname(os.path.abspath(__file__)) + ("/char_outputs/%s" % data_name)
    if os.path.exists(file_path):
        print("data already processed")
        return

    os.makedirs(file_path)

    make_char_1hot_matrix(splitted_data["x_test"],
                          splitted_data["y_test"],
                          "test",
                          file_path)

    make_char_1hot_matrix(splitted_data["x_train"],
                          splitted_data["y_train"],
                          "train",
                          file_path)

def load_data_from_file(name):
    path = os.path.dirname(os.path.abspath(__file__)) + "/char_outputs/" + name
    # check if folder exists
    if not os.path.exists(path):
        data = preprocess.load_from_file(name)
        preprocess_char_cnn(data, name)

    x_train = np.load(path + "/train_data.npy")
    x_test = np.load(path + "/test_data.npy")

    y_train = np.load(path + "/train_labels.npy")
    y_test = np.load(path + "/test_labels.npy")

    print("\nData Summary:")
    def count_positive(y_data):
        count = 0
        for y in y_data:
            if y == 1:
                count +=1
        return count, count / len(y_data)

    print("Train: Total Positive Labels=%s (%.4f)" % count_positive(y_train))
    print("Test: Total Positive Labels=%s (%.4f)" % count_positive(y_test))


    return x_train, y_train, x_test, y_test
