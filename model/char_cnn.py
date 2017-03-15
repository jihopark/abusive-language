#!/usr/bin/env python
"""
CharCNN Model from
Zhang, X.; Zhao, J.; and LeCun, Y. 2015. Character-level Convolutional
Networks for Text Classification. In Proceedings of NIPS.

also referenced https://github.com/johnb30/py_crepe
"""

import tensorflow as tf
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

class CharCNN(object):
    N_FILTERS = {"small": 256, "large": 1024}
    FILTER_KERNELS = [7, 7, 3, 3, 3, 3]
    POOL_SIZE = 3
    FULLY_CONNECTED_OUTPUT = {"small": 1024, "large": 2048}

    def __init__(self, name, vocab_size, text_len, n_classes,
            learning_rate=0.01, model_size="small"):
        print("Building Character CNN graph of name " + name)
        self.name = name
        self.vocab_size = vocab_size
        self.text_len = text_len

        input_shape = (vocab_size, text_len)

        tf.reset_default_graph()

        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32, [None, text_len, vocab_size])
            self.labels = tf.placeholder(tf.float32, [None, 1])

        with tf.name_scope("model"):
            self.layers = Sequential()
            with tf.name_scope("cnn-layer-0"):
                self.layers.add(Convolution1D(self.N_FILTERS[model_size],
                                              kernel_size=self.FILTER_KERNELS[0],
                                              padding='valid',
                                              activation='relu',
                                              input_shape=(text_len, vocab_size)))
                self.layers.add(MaxPooling1D(pool_size=self.POOL_SIZE))

            with tf.name_scope("cnn-layer-1"):
                self.layers.add(Convolution1D(self.N_FILTERS[model_size],
                                              kernel_size=self.FILTER_KERNELS[1],
                                              padding='valid',
                                              activation='relu'))
                self.layers.add(MaxPooling1D(pool_size=self.POOL_SIZE))

            with tf.name_scope("cnn-layer-2"):
                self.layers.add(Convolution1D(self.N_FILTERS[model_size],
                                              kernel_size=self.FILTER_KERNELS[2],
                                              padding='valid',
                                              activation='relu'))

            with tf.name_scope("cnn-layer-3"):
                self.layers.add(Convolution1D(self.N_FILTERS[model_size],
                                              kernel_size=self.FILTER_KERNELS[3],
                                              padding='valid',
                                              activation='relu'))

            with tf.name_scope("cnn-layer-4"):
                self.layers.add(Convolution1D(self.N_FILTERS[model_size],
                                              kernel_size=self.FILTER_KERNELS[4],
                                              padding='valid',
                                              activation='relu'))

            with tf.name_scope("cnn-layer-5"):
                self.layers.add(Convolution1D(self.N_FILTERS[model_size],
                                              kernel_size=self.FILTER_KERNELS[5],
                                              padding='valid',
                                              activation='relu'))
                self.layers.add(MaxPooling1D(pool_size=self.POOL_SIZE))

            with tf.name_scope("fully-connected-layer-0"):
                self.layers.add(Flatten())
                self.layers.add(Dense(self.FULLY_CONNECTED_OUTPUT[model_size],
                                      activation='relu'))
                self.layers.add(Dropout(0.5))

            with tf.name_scope("fully-connected-layer-1"):
                self.layers.add(Dense(self.FULLY_CONNECTED_OUTPUT[model_size],
                                      activation='relu'))
                self.layers.add(Dropout(0.5))

            with tf.name_scope("softmax"):
                self.layers.add(Dense(n_classes,
                                      activation='softmax'))

            with tf.name_scope("logits"):
                self.logits = self.layers(self.X, name="softmax_output")

            with tf.name_scope("training"):
                self.cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                self.train_op = self.optimizer.minimize(self.cross_entropy,
                        name="train_op")

            with tf.name_scope("prediction"):
                self.prediction = tf.argmax(self.logits, 1, name="prediction")
                self.accuracy = tf.reduce_mean(
                                tf.cast(tf.equal(self.prediction, self.labels), tf.float32), name="accuracy")
