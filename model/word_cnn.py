#!/usr/bin/env python
"""
WordCNN Model from
Kim Y. 2014. Convolutional Neural Network for
Sentence Classification In Proceedings of EMNLP.
"""

import tensorflow as tf


class WordCNN(object):
    def __init__(
            self, name, sequence_length, n_classes, vocab_size,
            filter_sizes, num_filters,
            embedding_size=300,
            l2_reg_lambda=0.0, embedding_static=False,
            word2vec_multi=False, learning_rate=0.001, dictionary=None):
        self.name = name
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.dictionary = dictionary

        # Placeholders for input, output and dropout
        with tf.name_scope("input"):
            self.X = tf.placeholder(
                tf.int32, [None, sequence_length], name="X")
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
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=not embedding_static,
                name="W")
            self.embedded_words = tf.nn.embedding_lookup(self.W, self.X)
            # Add channel dimension to make it [None, sequence_length,
            # embedding_size, 1]
            self.embedded_words_expanded = tf.expand_dims(
                self.embedded_words, -1)

            if word2vec_multi:
                self.W2 = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W2")
                self.embedded_words_2 = tf.nn.embedding_lookup(
                    self.W2, self.X)
                # Add channel dimension to make it [None, sequence_length,
                # embedding_size, 1]
                self.embedded_words_expanded_2 = tf.expand_dims(
                    self.embedded_words, -1)

        # Create a convolution + maxpool layer for each filter size
        channels = [self.embedded_words_expanded]
        if word2vec_multi:
            channels.append(self.embedded_words_expanded_2)

        pooled_outputs = []

        for n, channel in enumerate(channels):
            for filter_size in filter_sizes:
                with tf.name_scope("channel%s-conv-maxpool-%s" % (n, filter_size)):
                    # Convolution Layer
                    # filter_shape = [filter height, filter width, in_channels,
                    # out_channels]
                    filter_shape = [filter_size,
                                    embedding_size, 1, num_filters]
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
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes) * len(channels)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

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
