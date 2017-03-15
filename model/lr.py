#!/usr/bin/env python
"""Multivariable Linear Regression Model"""

import tensorflow as tf
import numpy as np

class LinearRegression(object):
    def __init__(self, n_dim, name, learning_rate=0.01):
        print("Building Linear Regression graph of name " + name)
        self.name = name
        self.n_dim = n_dim

        tf.reset_default_graph()
        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32,[None,n_dim])
            self.labels = tf.placeholder(tf.float32,[None,1])

        with tf.name_scope("weight"):
            self.W = tf.Variable(tf.random_uniform([n_dim, 1], -1, 1))

        with tf.name_scope("logits"):
            self.h = tf.matmul(self.X, self.W, name="apply_weights")
            self.sig_h = tf.nn.sigmoid(self.h, name="sigmoid")

        with tf.name_scope("training"):
            self.cost = -tf.reduce_mean(self.labels*tf.log(self.sig_h) +
                                        (1 - self.labels)*tf.log(1 - self.sig_h),
                                        name="cost")
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.train_op = self.optimizer.minimize(self.cost, name="train_op")
            tf.summary.scalar('cost', self.cost)

        with tf.name_scope("prediction"):
            self.prediction = tf.round(self.sig_h, name="prediction")
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.prediction, self.labels), tf.float32), name="accuracy")
            tf.summary.scalar('accuracy', self.accuracy)

        self.merge_summary = tf.summary.merge_all()
