import string
import re

def to_words(inputStr, padStart=0, padEnd=0, includePunct=True):
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

def to_chars(inputStr, includeSpace=False, includePunct=True):
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
    print(to_chars(char))
    print('end...')
