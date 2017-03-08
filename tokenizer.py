import string
import re


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
