import itertools
from tqdm import tqdm
from collections import Counter
from nltk import ngrams
from preprocess import concat_unshared_task_datasets as load_data
from tokenizer import to_words, to_chars

def ngram_counter(tokens, n, separator="_"):
    grams = []
    for i in range(1, n+1):
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

# def to_ngrams_rows(line, word2index):


def turn_into_word_ngram_matrix(datalist, n=2, vocabulary_size=10000):
    data = list(itertools.chain(*datalist))
    print(data[0])
    print(data[-1])

    print("\nTokenized texts into words")
    data = list(map(lambda x: to_words(x.lower()), data))
    print(data[0])
    print(data[-1])

    print("\nNgram-count the tokens")
    data = list(map(lambda x: ngram_counter(x, 2), data))
    print(data[0])
    print(data[-1])

    print("\nCreate dictionary")
    grams = Counter()
    for i in tqdm(range(len(data))):
        grams += data[i]
    print("total ngrams: %s" % len(grams))

    if len(grams) < vocabulary_size:
        vocabulary_size = len(grams)

    print("Most common:")
    for g, count in grams.most_common(10):
        print("%s: %7d" % (g, count))

    index2word, word2index, index2freq = get_dictionaries(grams, vocabulary_size)

    print("index2word: %s" % index2word[:10])
    print("index2freq: %s" % index2freq[:10])




if __name__ == '__main__':
    data = load_data()
    turn_into_word_ngram_matrix([data["racism"], data["none"]])
