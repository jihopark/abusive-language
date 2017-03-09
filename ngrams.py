import itertools
from collections import Counter
import os
import argparse

import pickle
import numpy as np
from nltk import ngrams
from tqdm import tqdm
from preprocess import concat_unshared_task_datasets as load_data
from tokenizer import to_words, to_chars

parser = argparse.ArgumentParser()
parser.add_argument('--nohup', action="store_true", help='turn this option when using with nohup')
args = vars(parser.parse_args())



# tokens: array of tokens, n: maximum ngram, start_n: minimum ngram,
# separator: char between each tokens
# returns ngrams (start_n ~ n)
def ngram_counter(tokens, n, start_n=1, separator="_"):
    grams = []
    for i in range(start_n, n+1):
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

def make_ngram_matrix(datalist, data_name, word_or_char="word",
        tokenizer_options={}, n=2, vocab_size=100000):
    print("\n\nData Name:" + data_name)
    # check if already there
    file_path = os.path.abspath("data/ngram/%s" % data_name)
    if os.path.exists(file_path):
        print("data already processed")
        return

    data = list(itertools.chain(*datalist))
    print(data[0])
    print(data[-1])

    print("\nTokenize texts")
    tokenizer = to_words if word_or_char == "word" else to_chars
    data = [tokenizer(x.lower(), tokenizer_options) for x in data]

    print(data[0])
    print(data[-1])

    print("\nNgram-count the tokens")
    data = list(map(lambda x: ngram_counter(x, n), data))
    print(data[0])
    print(data[-1])

    print("\nCreate dictionary")
    grams = Counter()

    # do not output tqdm progress if nohup
    range_ = range(len(data))
    if not args["nohup"]:
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
    print(data.shape)

    print("\nSave np array & metadata into file at %s" % file_path)

    os.makedirs(file_path)

    np.save(file_path + "/data.npy", data)

    metadata = {
        "index2word": index2word,
        "index2freq": index2freq,
        "word2index": word2index
        }
    with open(file_path + "/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

if __name__ == '__main__':
    data = load_data()
    # racism word ngram
    make_ngram_matrix([data["racism"], data["none"]], n=3,
            data_name="racism_binary_word_pad_none_3gram")
    make_ngram_matrix([data["racism"], data["none"]], n=4,
            data_name="racism_binary_word_pad_1_4gram")
    make_ngram_matrix([data["racism"], data["none"]], n=3,
            data_name="racism_binary_word_pad_none_3gram_remove_punc",
            tokenizer_options={"includePunct":False})
    make_ngram_matrix([data["racism"], data["none"]], n=4,
            data_name="racism_binary_word_pad_1_4gram_remove_punc",
            tokenizer_options={"includePunct":False})

    # sexism word ngram
    make_ngram_matrix([data["sexism"], data["none"]], n=3,
            data_name="sexism_binary_word_pad_none_3gram")
    make_ngram_matrix([data["sexism"], data["none"]], n=4,
            data_name="sexism_binary_word_pad_1_4gram")
    make_ngram_matrix([data["sexism"], data["none"]], n=3,
            data_name="sexism_binary_word_pad_none_3gram_remove_punc",
            tokenizer_options={"includePunct":False})
    make_ngram_matrix([data["sexism"], data["none"]], n=4,
            data_name="sexism_binary_word_pad_1_4gram_remove_punc",
            tokenizer_options={"includePunct":False})

    # racism char ngram
    make_ngram_matrix([data["racism"], data["none"]], n=3,
            word_or_char="char",
            data_name="racism_binary_char_pad_none_3gram")
    make_ngram_matrix([data["racism"], data["none"]], n=4,
            word_or_char="char",
            data_name="racism_binary_char_pad_1_4gram")
    make_ngram_matrix([data["racism"], data["none"]], n=3,
            word_or_char="char",
            data_name="racism_binary_char_pad_none_3gram_remove_punc",
            tokenizer_options={"includePunct":False})
    make_ngram_matrix([data["racism"], data["none"]], n=4,
            word_or_char="char",
            data_name="racism_binary_char_pad_1_4gram_remove_punc",
            tokenizer_options={"includePunct":False})

    # sexism char ngram
    make_ngram_matrix([data["sexism"], data["none"]], n=3,
            word_or_char="char",
            data_name="sexism_binary_char_pad_none_3gram")
    make_ngram_matrix([data["sexism"], data["none"]], n=4,
            word_or_char="char",
            data_name="sexism_binary_char_pad_1_4gram")
    make_ngram_matrix([data["sexism"], data["none"]], n=3,
            word_or_char="char",
            data_name="sexism_binary_char_pad_none_3gram_remove_punc",
            tokenizer_options={"includePunct":False})
    make_ngram_matrix([data["sexism"], data["none"]], n=4,
            word_or_char="char",
            data_name="sexism_binary_char_pad_1_4gram_remove_punc",
            tokenizer_options={"includePunct":False})

