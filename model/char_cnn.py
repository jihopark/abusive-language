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
from keras.objectives import categorical_crossentropy
from keras import initializers
from keras.regularizers import l2

class CharCNN(object):
    N_FILTERS = {"small": 256, "large": 1024}
    FILTER_KERNELS = [7, 7, 3, 3, 3, 3]
    POOL_SIZE = 3
    FULLY_CONNECTED_OUTPUT = {"small": 1024, "large": 2048}
    INIT_VAR = {"small": 0.05, "large": 0.02}

    def __init__(self, name, vocab_size, text_len, n_classes,
            learning_rate=0.01, model_size="small", model_depth="shallow",
            positive_weight=1, fully_connected_l2=0, cnn_l2=0):
        print("Building Character CNN graph of name " + name)
        self.name = name
        self.vocab_size = vocab_size
        self.text_len = text_len

        input_shape = (vocab_size, text_len)

        tf.reset_default_graph()

        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32, [None, text_len, vocab_size])
            self.labels = tf.placeholder(tf.int64, [None, 1])
            self.labels_one_hot = tf.cast(tf.reshape(tf.one_hot(self.labels,
                    depth=n_classes), [-1, 2]), dtype="float32")

        with tf.name_scope("nn-layers"):
            self.layers = Sequential()
            with tf.name_scope("cnn-layer-0"):
                self.layers.add(Convolution1D(self.N_FILTERS[model_size],
                                              kernel_size=self.FILTER_KERNELS[0],
                                              padding='valid',
                                              kernel_initializer=
                                                  initializers.random_normal(
                                                      stddev=self.INIT_VAR[model_size]),
                                              activation='relu',
                                              kernel_regularizer=l2(cnn_l2),
                                              input_shape=(text_len, vocab_size)))
                self.layers.add(MaxPooling1D(pool_size=self.POOL_SIZE))

            with tf.name_scope("cnn-layer-1"):
                self.layers.add(Convolution1D(self.N_FILTERS[model_size],
                                              kernel_size=self.FILTER_KERNELS[1],
                                              padding='valid',
                                              kernel_initializer=
                                                initializers.random_normal(
                                                    stddev=self.INIT_VAR[model_size]),
                                              kernel_regularizer=l2(cnn_l2),
                                              activation='relu'))
                self.layers.add(MaxPooling1D(pool_size=self.POOL_SIZE))

            if model_depth == "deep":
                with tf.name_scope("cnn-layer-2"):
                    self.layers.add(Convolution1D(self.N_FILTERS[model_size],
                                                  kernel_size=self.FILTER_KERNELS[2],
                                                  padding='valid',
                                                  kernel_initializer=
                                                    initializers.random_normal(
                                                        stddev=self.INIT_VAR[model_size]),
                                                  kernel_regularizer=l2(cnn_l2),
                                                  activation='relu'))

                with tf.name_scope("cnn-layer-3"):
                    self.layers.add(Convolution1D(self.N_FILTERS[model_size],
                                                  kernel_size=self.FILTER_KERNELS[3],
                                                  padding='valid',
                                                  kernel_initializer=
                                                    initializers.random_normal(
                                                        stddev=self.INIT_VAR[model_size]),
                                                  kernel_regularizer=l2(cnn_l2),
                                                  activation='relu'))

                with tf.name_scope("cnn-layer-4"):
                    self.layers.add(Convolution1D(self.N_FILTERS[model_size],
                                                  kernel_size=self.FILTER_KERNELS[4],
                                                  padding='valid',
                                                  kernel_initializer=
                                                    initializers.random_normal(
                                                        stddev=self.INIT_VAR[model_size]),
                                                  kernel_regularizer=l2(cnn_l2),
                                                  activation='relu'))

                with tf.name_scope("cnn-layer-5"):
                    self.layers.add(Convolution1D(self.N_FILTERS[model_size],
                                                  kernel_size=self.FILTER_KERNELS[5],
                                                  padding='valid',
                                                  kernel_initializer=
                                                    initializers.random_normal(
                                                        stddev=self.INIT_VAR[model_size]),
                                                  kernel_regularizer=l2(cnn_l2),
                                                  activation='relu'))
                    self.layers.add(MaxPooling1D(pool_size=self.POOL_SIZE))



            with tf.name_scope("fully-connected-layer-0"):
                self.layers.add(Flatten())
                self.layers.add(Dense(self.FULLY_CONNECTED_OUTPUT[model_size],
                                      kernel_initializer=
                                        initializers.random_normal(
                                            stddev=self.INIT_VAR[model_size]),
                                      kernel_regularizer=l2(fully_connected_l2),
                                      activation='relu'))
                self.layers.add(Dropout(0.5))

            if model_depth == "deep":
                with tf.name_scope("fully-connected-layer-1"):
                    self.layers.add(Dense(self.FULLY_CONNECTED_OUTPUT[model_size],
                                          kernel_initializer=
                                            initializers.random_normal(
                                                stddev=self.INIT_VAR[model_size]),
                                          kernel_regularizer=l2(fully_connected_l2),
                                          activation='relu'))
                    self.layers.add(Dropout(0.5))

            with tf.name_scope("softmax"):
                self.layers.add(Dense(n_classes,
                                      kernel_initializer=
                                        initializers.random_normal(
                                            stddev=self.INIT_VAR[model_size]),
                                      activation='softmax'))

            with tf.name_scope("logits"):
                self.logits = self.layers(self.X)

            # print neural network(nn) layer details
            print("\nnn layers input shape:%s" % str(self.layers.input_shape))
            print("\nnn layers output shape:%s" %
                    str(self.layers.output_shape))
            print("\nnn layer option:size=%s, depth=%s" % (model_size,
                                                           model_depth))
            print("\nnn layers trainable weights length:")
            for i, w in enumerate(self.layers.trainable_weights):
                print("layer %s: %s" % (i, str(w.get_shape())))
            print("L2 regularization CNN layer=%.2f, FC layer=%.2f" % (cnn_l2,
                fully_connected_l2))

        with tf.name_scope("training"):
            # using weighted cross-entropy loss since we have imbalanced
            # dataset. put more emphasis on positive training examples
            print("\nUsing weighted cross-entropy loss with positive_weight=%s"
                  % positive_weight)
            self.positive_weight = tf.constant(positive_weight, dtype="float32")
            self.cost = tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(self.labels_one_hot,
                                                         self.logits, self.positive_weight),
                name="cross_entropy_loss")

            print("\nUsing Gradient DescentOptimizer with learning_rate=%s"
                  % learning_rate)
            self.optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate)
            self.train_op = self.optimizer.minimize(self.cost, name="train_op")
            tf.summary.scalar("cost", self.cost)

        with tf.name_scope("prediction"):
            self.prediction = tf.argmax(self.logits, 1, name="prediction")

        self.merge_summary = tf.summary.merge_all()
