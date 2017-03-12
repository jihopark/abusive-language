#!/usr/bin/env python
"""Multivariable Linear Regression Model"""

import tensorflow as tf
import numpy as np

class LinearRegression(object):
    def __init__(self, n_dim, name, learning_rate=0.1):
        print("Building Linear Regression graph of name " + name)
        self.name = name
        tf.reset_default_graph()
        # placeholders for data
        self.X = tf.placeholder(tf.float32,[None,n_dim])
        self.labels = tf.placeholder(tf.float32,[None,1])

        self.W = tf.Variable(tf.random_uniform([n_dim, 1], -1, 1))

        self.h = tf.matmul(self.X, self.W)
        self.sig_h = tf.div(1., 1. + tf.exp(-self.h)) # sigmoid function

        self.cost = -tf.reduce_mean(self.labels*tf.log(self.sig_h) +
                                    (1 - self.labels)*tf.log(1 - self.sig_h))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.train_op = self.optimizer.minimize(self.cost)
