#!/usr/bin/env python
"""
WordCNN Model from
Kim Y. 2014. Convolutional Neural Network for
Sentence Classification In Proceedings of EMNLP.
"""

from keras.layers import Input, Dense, Embedding, Convolution1D, MaxPooling1D
from keras.layers import Flatten, Concatenate, Dropout
from keras.models import Model
from keras.optimizers import Adam


class WordCNN(object):
    def __init__(
            self, sequence_length, n_classes, vocab_size,
            filter_sizes, num_filters,
            embedding_size=300,
            dropout_prob=0,
            embedding_static=False,
            learning_rate=0.001):
        inputs = Input(shape=(sequence_length,))
        embedding_layer = Embedding(input_dim=vocab_size,
                                    output_dim=embedding_size,
                                    trainable=(not embedding_static))(inputs)


        conv_blocks = []
        for filter_size in filter_sizes:
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=filter_size,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(embedding_layer)
            conv = MaxPooling1D(pool_size=sequence_length - filter_size + 1)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

        cnn = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        output = Dense(n_classes,
                            activation="softmax")(Dropout(dropout_prob)(cnn))
        self.model = Model(inputs, output)

        self.model.compile(loss="categorical_crossentropy",
                           optimizer=Adam(lr=learning_rate),
                           metrics=["accuracy"])
        self.model.summary()
