from collections import Counter
import os

import pickle
import numpy as np
from nltk import ngrams
from tqdm import tqdm
from . import tokenizer


# tokens: array of tokens, n: maximum ngram, start_n: minimum ngram,
# separator: char between each tokens
# returns ngrams (m ~ n)
def ngram_counter(tokens, n, m=1, separator="_"):
    grams = []
    for i in range(m, n+1):
        grams += list(ngrams(tokens, i))
    grams = map(lambda x: separator.join(x), grams)
    return Counter(grams)

def get_dictionaries(data, vocabulary_size):
    index2word = ["UNK"]
    word2index = {"UNK": 0}
    index2freq = [0]
    i = 1
    for g, count in data.most_common(vocabulary_size):
        index2word.append(g)
        word2index[g] = i
        index2freq.append(count)
        i += 1
    return index2word, word2index, index2freq

def make_ngram_matrix(data, labels, test_data, test_labels, data_name, word_or_char="word",
        tokenizer_options={}, n=2, m=1, vocab_size=50000, show_tqdm=False):
    print("\n\nData Name:" + data_name)
    # check if already there
    file_path = os.path.dirname(os.path.abspath(__file__)) + ("/ngram_outputs/%s" % data_name)
    if os.path.exists(file_path):
        print("data already processed")
        return

    print(data[0])
    print(data[-1])

    print("\nTokenize texts")
    _tokenizer = tokenizer.to_words if word_or_char == "word" else tokenizer.to_chars
    data = [_tokenizer(x.lower(), tokenizer_options) for x in data]
    test_data = [_tokenizer(x.lower(), tokenizer_options) for x in test_data]

    print(data[0])
    print(data[-1])

    print("\nNgram-count the tokens")
    data = list(map(lambda x: ngram_counter(x, n, m), data))
    test_data = list(map(lambda x: ngram_counter(x, n, m), test_data))

    print(data[0])
    print(data[-1])

    print("\nCreate dictionary")
    grams = Counter()

    # do not output tqdm progress if nohup
    range_ = range(len(data))
    if show_tqdm:
        range_ = tqdm(range_)

    for i in range_:
        grams += data[i]
    print("total ngrams: %s" % len(grams))

    if len(grams) < vocab_size:
        vocab_size = len(grams)

    print("Most common:")
    for g, count in grams.most_common(10):
        print("%s: %7d" % (g, count))

    index2word, word2index, index2freq = get_dictionaries(grams, vocab_size)

    print("index2word: %s" % index2word[:100])
    print("index2freq: %s" % index2freq[:100])

    print("\nTurn ngram-count to ngram-feature rows")
    for i, d in enumerate(data):
        row = np.zeros(vocab_size + 1)
        for g in list(d.elements()):
            if g in word2index:
                row[word2index[g]] = 1
        data[i] = row

    data = np.array(data)
    print(data[0])
    print(data[-1])

    #include labels in the data
    data = np.hstack((data, labels))
    print(data.shape)

    print("\n make test dataset into ngram-feature rows")
    for i, d in enumerate(test_data):
        row = np.zeros(vocab_size + 1)
        for g in list(d.elements()):
            if g in word2index:
                row[word2index[g]] = 1
        test_data[i] = row

    test_data = np.array(test_data)
    print(test_data[0])
    print(test_data[-1])

    #include labels in the data
    test_data = np.hstack((test_data, test_labels))
    print(test_data.shape)



    print("\nSave np array & metadata into file at %s" % file_path)

    os.makedirs(file_path)

    np.save(file_path + "/train_data.npy", data)
    np.save(file_path + "/test_data.npy", test_data)

    metadata = {
        "index2word": index2word,
        "index2freq": index2freq,
        "word2index": word2index
        }
    with open(file_path + "/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)


def load_data_from_file(name):
    path = os.path.dirname(os.path.abspath(__file__)) + "/ngram_outputs/" + name
    # check if folder exists
    if not os.path.exists(path):
        print("no dataset exists with the name %s at path %s" % (name, path))
        return None, None

    data_array = np.load(path + "/data.npy")
    with open(path + "/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return data_array, metadata

