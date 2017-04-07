#!/usr/bin/env python
"""
Hybrid version of WordCNN and CharCNN
"""

import tensorflow as tf
import numpy as np

class HybridCNN(object):
    def __init__(
            self, name, word_len, char_len,
            n_classes, word_vocab_size, char_vocab_size,
            word_filter_sizes,
            char_filter_sizes,
            num_filters,
            embedding_size=300,
            l2_reg_lambda=0.0, embedding_static=False,
            learning_rate=0.001):
        print("Building Hybrid Char-Word CNN graph of name " + name)
        print("learning rate=%s, l2_lambda=%s" % (learning_rate, l2_reg_lambda))
        self.name = name
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.word_len = word_len

        # Placeholders for input, output and dropout
        with tf.name_scope("input"):
            self.X_word = tf.placeholder(
                tf.int32, [None, word_len], name="X_word")
            self.X_char = tf.placeholder(
                tf.float32, [None, char_len, char_vocab_size], name="X_char")
            self.labels = tf.placeholder(
                tf.int64, [None, 1], name="labels")
            self.labels_one_hot = tf.cast(tf.reshape(tf.one_hot(self.labels,
                                                                depth=n_classes), [-1, 2]),
                                                                dtype="float32")

        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([word_vocab_size, embedding_size], -1.0, 1.0),
                trainable=not embedding_static,
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.X_word)
            # Add channel dimension to make it [None, word_len,
            # embedding_size, 1]
            self.embedded_chars_expanded = tf.expand_dims(
                self.embedded_chars, -1)

        # Expand dimension for char input as well and add it as a channel
        self.X_char_expanded = tf.expand_dims(self.X_char, -1)
        channels = [self.embedded_chars_expanded, self.X_char_expanded]

        print("\nWord feature shape=%s/Char feature shape=%s" %
                (str(self.embedded_chars_expanded.get_shape().as_list()),
                 str(self.X_char_expanded.get_shape().as_list())))


        pooled_outputs = []

        # Create a convolution + maxpool layer for each filter size
        # convolution output of all features
        assert len(word_filter_sizes) == len(char_filter_sizes)
        filter_size = np.vstack((word_filter_sizes, char_filter_sizes))
        for i in range(len(word_filter_sizes)):
            for n, channel in enumerate(channels):
                with tf.name_scope("channel%s-conv-maxpool-%s" % (n, filter_size[n][i])):
                    # Convolution Layer
                    # filter_shape = [filter height, filter width, in_channels,
                    # out_channels]

                    filter_width = embedding_size if n == 0 else char_vocab_size
                    filter_shape = [filter_size[n][i],
                                    filter_width, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(
                        filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(
                        0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        channel,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    length = word_len if n == 0 else char_len
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, length - filter_size[n][i] + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(word_filter_sizes) * len(channels)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print("\nAfter Concat Pooled Shape=%s" %
                str(self.h_pool_flat.get_shape()))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(
                self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) logits and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, n_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[n_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.prediction = tf.argmax(self.logits, 1, name="prediction")

        # CalculateMean cross-entropy loss
        with tf.name_scope("training"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels_one_hot)
            self.cost = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            tf.summary.scalar("cost", self.cost)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = self.optimizer.compute_gradients(self.cost)
            self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        self.merge_summary = tf.summary.merge_all()
