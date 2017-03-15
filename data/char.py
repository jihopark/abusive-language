#!/usr/bin/env python
""" Helper functions for character-based features"""

import os

import numpy as np
from . import tokenizer

TWEET_MAX_LEN = 140
FEATURES = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’’’/\|_@#$%ˆ&*~‘+-=<>()[]{}")
N_DIM = len(FEATURES)

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
    np.save(file_path + "/%s_labels.npy" % split_name, labels)
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

    make_char_1hot_matrix(splitted_data["x_valid"],
                          splitted_data["y_valid"],
                          "valid",
                          file_path)

    make_char_1hot_matrix(splitted_data["x_train"],
                          splitted_data["y_train"],
                          "train",
                          file_path)

def load_data_from_file(name):
    path = os.path.dirname(os.path.abspath(__file__)) + "/char_outputs/" + name
    # check if folder exists
    if not os.path.exists(path):
        print("no dataset exists with the name %s at path %s" % (name, path))
        return None, None

    x_train = np.load(path + "/train_data.npy")
    x_valid = np.load(path + "/valid_data.npy")
    x_test = np.load(path + "/test_data.npy")

    y_train = np.load(path + "/train_labels.npy")
    y_valid = np.load(path + "/valid_labels.npy")
    y_test = np.load(path + "/test_labels.npy")

   return x_train, y_train, x_valid, y_valid, x_test, y_test
