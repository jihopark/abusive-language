#!/usr/bin/env python
"""
Hybrid version of WordCNN and CharCNN
"""
from keras.layers import Input, Dense, Embedding, Convolution1D, MaxPooling1D
from keras.layers import Flatten, Concatenate, Dropout
from keras.models import Model
from keras.optimizers import Adam


class HybridCNN(object):
    def __init__(
            self, word_len, char_len,
            n_classes, word_vocab_size, char_vocab_size,
            word_filter_sizes,
            char_filter_sizes,
            num_filters,
            embedding_size=300,
            dropout_prob=0,
            embedding_static=False,
            learning_rate=0.001):

        input_word = Input(shape=(word_len,))
        input_char = Input(shape=(char_len, char_vocab_size))

        embedding_layer = Embedding(input_dim=word_vocab_size,
                                    output_dim=embedding_size,
                                    trainable=(not embedding_static)
                                    )(input_word)
        # word cnn layers
        conv_blocks = []
        for filter_size in word_filter_sizes:
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=filter_size,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(embedding_layer)
            conv = MaxPooling1D(pool_size=word_len - filter_size + 1)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

        # char cnn layers
        for filter_size in char_filter_sizes:
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=filter_size,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(input_char)
            conv = MaxPooling1D(pool_size=char_len - filter_size + 1)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

        cnn = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        output = Dense(n_classes,
                       activation="softmax")(Dropout(dropout_prob)(cnn))
        self.model = Model(inputs=[input_char, input_word],
                           outputs=output)

        self.model.compile(loss="categorical_crossentropy",
                           optimizer=Adam(lr=learning_rate),
                           metrics=["accuracy"])
        self.model.summary()
