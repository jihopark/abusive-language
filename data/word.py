#!/usr/bin/env python
""" Preprocessing script for WordCNN"""

import os
import numpy as np
import gensim

from tensorflow.contrib import learn
from . import preprocess
from . import tokenizer

def one_hot_to_words(row, dictionary):
    if dictionary == None:
        return []
    return dictionary.reverse([row])

def create_vocabulary_dataset(word2vec_model, data):
    x_tokenized = list(map(lambda x: tokenizer.tokenize_with_dictionary(x,
        word2vec_model), data))
    print("Tokenizing tweets with tokenize_for_word2vec\n")
    print(x_tokenized[:10])
    max_document_length = max([len(x) for x in x_tokenized])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency=2)
    x_joined = list(map(lambda x: " ".join(x), x_tokenized))
    x_vocab = np.array(list(vocab_processor.fit_transform(x_joined)))
    print("changed data %s into %s" % (str(data.shape), str(x_vocab.shape)))
    return vocab_processor, max_document_length, x_vocab

def vocabulary_into_wordvec_embeddings(word2vec_model, vocab):
    reverse_dict = dict()
    for key in vocab.vocabulary_._mapping.keys():
        value = vocab.vocabulary_._mapping[key]
        reverse_dict[value] = key

    word2vec_vectors = []
    for i in range(len(vocab.vocabulary_)):
        if reverse_dict[i] in word2vec_model:
            word2vec_vectors.append(word2vec_model[reverse_dict[i]])
        else:
            word2vec_vectors.append(np.zeros(300))
    return np.array(word2vec_vectors)

def fit_input_into_vocab(input, vocab):
    x_tokenized = tokenizer.tokenize_with_dictionary(input, vocab=vocab.vocabulary_._mapping.keys())
    print("Tokenizing tweets with tokenize_for_dictionary\n")
    print(x_tokenized)
    x_joined = [" ".join(x_tokenized)]
    x_vocab = np.array(list(vocab.fit_transform(x_joined)))
    print("changed data %s into %s" % (len(input), str(x_vocab.shape)))
    return x_vocab


def save_word_cnn(data, data_name):
    print("loading pretrained embedding")
    pretrained_word2vec = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin',
                                                                       binary=True)
     # check if already there
    file_path = os.path.dirname(os.path.abspath(__file__)) + ("/word_outputs/%s" % data_name)
    if os.path.exists(file_path):
        print("data already processed")
        return
    os.makedirs(file_path)

    vocab, max_len, x_vocab = create_vocabulary_dataset(pretrained_word2vec,
                                                        np.concatenate([data["x_train"],
                                                                        data["x_test"]]))
    init_W = vocabulary_into_wordvec_embeddings(pretrained_word2vec, vocab)

    x_train = x_vocab[:len(data["x_train"])]
    x_test = x_vocab[len(data["x_test"])*-1:]

    assert len(x_train) == len(data["y_train"])
    assert len(x_test) == len(data["y_test"])

    np.save(file_path + "/initW.npy", init_W)
    vocab.save(file_path + "/metadata.vocab")
    np.save(file_path + "/x_train.npy", x_train)
    np.save(file_path + "/x_test.npy", x_test)
    np.save(file_path + "/y_train.npy",
            data["y_train"].reshape(len(x_train), 1))
    np.save(file_path + "/y_test.npy",
            data["y_test"].reshape(len(x_test), 1))


def load_data_from_file(data_name):
    file_path = os.path.dirname(os.path.abspath(__file__)) + ("/word_outputs/%s" % data_name)
    if not os.path.exists(file_path):
        save_word_cnn(preprocess.load_from_file(data_name), data_name)
    return (
        np.load(file_path + "/x_train.npy"),
        np.load(file_path + "/y_train.npy"),
        np.load(file_path + "/x_test.npy"),
        np.load(file_path + "/y_test.npy"),
        np.load(file_path + "/initW.npy"),
        learn.preprocessing.VocabularyProcessor.restore(file_path + "/metadata.vocab")
        )
