import string
import re

def word_filtering(inputStr, includePunct, includeSpace=True):
    whitelistStr = string.ascii_letters + string.digits
    whitelistStr = (whitelistStr + string.punctuation) if includePunct else whitelistStr
    whitelistStr = (whitelistStr + string.whitespace) if includeSpace else whitelistStr

    whitelist = set(whitelistStr)
    processedStr = ''.join(filter(whitelist.__contains__, inputStr))

    return processedStr

# to_words
# options = {"padStart": integer, "padEnd": integer, "includePunct":boolean, "filterNonAlphaNumPun":boolean}
def to_words(inputStr, options={}):
    includePunct = True if "includePunct" not in options else options["includePunct"]
    filterNonAlphaNumPun = True if "filterNonAlphaNumPun" not in options else options["filterNonAlphaNumPun"]
    padStart = 0 if "padStart" not in options else options["padStart"]
    padEnd = 0 if "padEnd" not in options else options["padEnd"]

    word_list = [];
    #insert <start> tokens if any
    word_list.extend(['<start>'] * padStart)

    processedInputStr = ''

    # filter non-alpha-num
    if filterNonAlphaNumPun:
        processedInputStr = word_filtering(inputStr, includePunct)
    else:
        processedInputStr = inputStr

    # space-splitting punctuation
    if includePunct:
        processedInputStr = re.sub("([.,!'?()])", r' \1 ', processedInputStr)

    word_list += processedInputStr.split()

    #insert <end> tokens if any
    word_list.extend(['<end>'] * padEnd)

    return word_list

# to_chars
# options = {"includeSpace": boolean, "includePunct": boolean, "filterNonAlphaNumPun":boolean}
def to_chars(inputStr, options={}):
    filterNonAlphaNumPun = True if "filterNonAlphaNumPun" not in options else options["filterNonAlphaNumPun"]
    includePunct = True if "includePunct" not in options else options["includePunct"]
    includeSpace = False if "includeSpace" not in options else options["includeSpace"]

    char_list = [];

    # filter non-alpha-num & handle punc
    if filterNonAlphaNumPun:
        inputStr = word_filtering(inputStr, includePunct)
    else:
        inputStr = inputStr if includePunct else inputStr.translate(str.maketrans('', '', string.punctuation))

    word_tokens = inputStr.split()
    for word_token in word_tokens:
        token_list = list(word_token)
        char_list += ([" "] + token_list) if includeSpace else token_list

    return char_list

# testing
if __name__ == '__main__':
    word = "This is python testing on jupyter. This is Nayeon's Testing page? hoho! Haha!!!" + u'\U0001f602'
    char = 'anaconda is nice!!! Wow. a' + u'\U0001f602'

    print(word)

    # print(to_words(word))
    # print(to_words(word, options={"filterNonAlphaNumPun": False }))
    # print(to_words(word, options={"padStart": 1, "padEnd": 1}))
    # print(to_chars(char))
    # print(to_chars(char, options={"filterNonAlphaNumPun": False}))
    # print(to_chars(char, options={"includePunct": False}))

    print(to_words(word, options={"includePunct":True}))
    print(to_words(word, options={"includePunct":False}))
    print(to_words(word, options={"includePunct":True, "filterNonAlphaNumPun": False}))
    print(to_words(word, options={"includePunct":False, "filterNonAlphaNumPun": False}))
    print(to_chars(char, options={"filterNonAlphaNumPun": True, "includeSpace": False, "includePunct":True}))
    print(to_chars(char, options={"filterNonAlphaNumPun": True, "includeSpace": False, "includePunct":False}))
    print(to_chars(char, options={"filterNonAlphaNumPun": True, "includeSpace": False, "includePunct":True}))
    print(to_chars(char, options={"filterNonAlphaNumPun": True, "includeSpace": True, "includePunct":True}))
    print('end...')
