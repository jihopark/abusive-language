import string
import re

#filter_non_alphanum_punc
def filter_non_alphanum_punc(inputStr):
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', inputStr)

# to_words
# options = {"padStart": integer, "padEnd": integer, "includePunct":boolean, "filterNonAlphaNumPun":boolean}
def to_words(inputStr, options={}):
    includePunct = True if "includePunct" not in options else options["includePunct"]
    filterNonAlphaNumPun = True if "filterNonAlphaNumPun" not in options else options["filterNonAlphaNumPun"]
    padStart = 0 if "padStart" not in options else options["padStart"]
    padEnd = 0 if "padEnd" not in options else options["padEnd"]

    word_list = [];

    #append <start> tokens if any
    word_list.extend(['<start>'] * padStart)

    # apply additional filtering to inputStr if necessary
    if filterNonAlphaNumPun:
        inputStr = filter_non_alphanum_punc(inputStr)

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
# options = {"includeSpace": boolean, "includePunct": boolean, "filterNonAlphaNumPun":boolean}
def to_chars(inputStr, options={}):
    filterNonAlphaNumPun = True if "filterNonAlphaNumPun" not in options else options["filterNonAlphaNumPun"]
    includePunct = True if "includePunct" not in options else options["includePunct"]
    includeSpace = False if "includeSpace" not in options else options["includeSpace"]

    char_list = [];

    if filterNonAlphaNumPun:
        inputStr = filter_non_alphanum_punc(inputStr)

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
    word = "This is python testing on jupyter. This is Nayeon's Testing page? hoho! Haha!!!" + u'\U0001f602'
    char = 'anaconda is nice!!! Wow. a' + u'\U0001f602'

    print(word)

    print(to_words(word))
    print(to_words(word, options={"filterNonAlphaNumPun": False }))
    print(to_words(word, options={"padStart": 1, "padEnd": 1}))
    print(to_chars(char))
    print(to_chars(char, options={"filterNonAlphaNumPun": False}))
    print(to_chars(char, options={"includePunct": False}))
    print('end...')
