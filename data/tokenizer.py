import string
import re

from wordsegment import segment
from nltk.tokenize import TweetTokenizer
import enchant

tknzr = TweetTokenizer(preserve_case=False, reduce_len=True)
eng_dictionary = enchant.Dict("en_US")

# tokenize words with using dictionary to segment further
def tokenize_with_dictionary(tweet, vocab=None, word2vec=None, doSegment=True):
    words = tknzr.tokenize(tweet)
    tokens = []
    in_dictionary = []

    # first check if in dictionary
    for word in words:
        tokens.append(word)
        if vocab:
            is_in_dictionary = word in vocab
        elif word2vec:
            is_in_dictionary = word in word2vec
        else:
            is_in_dictionary = eng_dictionary.check(word)
        in_dictionary.append(is_in_dictionary)

    # if not segment the word ex.himan => hi, man
    final_tokens = []
    for i, token in enumerate(tokens):
        if in_dictionary[i]:
            final_tokens.append(token)
        else:
            if doSegment:
                segments = segment(token)
                final_tokens += segments
            else:
                final_tokens.append(token)
    return final_tokens

# to_words
# options = {"padStart": integer, "padEnd": integer, "includePunct":boolean}
def to_words(inputStr, options={}):
    includePunct = True if "includePunct" not in options else options["includePunct"]
    padStart = 0 if "padStart" not in options else options["padStart"]
    padEnd = 0 if "padEnd" not in options else options["padEnd"]

    word_list = [];

    #append <start> tokens if any
    word_list.extend(['<start>'] * padStart)

    processedInputStr = ''
    if includePunct:
        processedInputStr = re.sub("([.,!'?()])", r' \1 ', inputStr)
    else:
        processedInputStr = inputStr.translate(str.maketrans('', '', string.punctuation))

    word_list += processedInputStr.split()

    #append <end> tokens if any
    word_list.extend(['<end>'] * padEnd)

    return word_list

# to_chars
# options = {"includeSpace": boolean, "includePunct": boolean}
def to_chars(inputStr, options={}):
    includePunct = True if "includePunct" not in options else options["includePunct"]
    includeSpace = False if "includeSpace" not in options else options["includeSpace"]

    char_list = [];

    puncHandledStr = inputStr
    if not includePunct:
        puncHandledStr = inputStr.translate(str.maketrans('', '', string.punctuation))

    word_tokens = puncHandledStr.split()

    for word_token in word_tokens:
            processed = " ".join(word_token)
            char_list += processed.split()
            char_list += " "

    del char_list[-1]
    return char_list

# testing
if __name__ == '__main__':
    word = "This is python testing on jupyter. This is Nayeon's Testing page? hoho! Haha!!!"
    char = 'anaconda is nice!!! Wow. a'

    print(to_words(word))
    print(to_words(word, options={"padStart": 1, "padEnd": 1}))
    print(to_chars(char))
    print(to_chars(char, options={"includePunct": False}))
    print('end...')
