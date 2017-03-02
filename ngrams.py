# PREPROCESS.PY(@JIHO)
# 	preprocess_tweet(tweet)
# 		Args:
# 			tweet: string
# 		Returns:
# 			String

# TOKENIZE.PY (@NAYEON)
# 	to_words(padStart=0, padEnd=0, includePunct=True)
# 		Args:
# 			padStart: integer (number of <start> tokens at the front)
# 			padEnd: integer (number of <end> token at the end)
# 			includePunct: boolean (whether to include punctuation in the token)
# 		Returns:
# 			A list of word tokens
# 	to_chars(includeSpace=False, includePunct=True)
# 		Args:
# 			includeSpace: boolean
# 			includePunct: boolean (whether to include punctuation in the token)
# 		Returns
# 			A list of char tokens

# NGRAM.PY(@JAY)
# 	covert_to_ngrams(data)
# 		Args:
# 			data: list of strings
# 		Returns:
# 			dictionary (key: n-gram, value: index)
# 			data converted into matrix (np.matrix)

import numpy as np
import pandas as pd
from preprocess import preprocess_tweet
from collections import OrderedDict

# tokens : list of words or characters in a string
def convert_to_ngrams(tokens, n=1, seperator=''):
	ngrams = OrderedDict({})

	# cap n-grams to length of tokens
	if n>len(tokens):
		n = len(tokens)

	# For each gram
	index = 0
	for i in range(1,n+1):
		for j in range (0,len(tokens)):
			# group char/words together based on i-gram
			if i > 1:
				for k in range(2,i+1):
					if j+k-1 < len(tokens):
						joined_token = seperator.join(tokens[j:j+k])
						if joined_token not in ngrams.keys():
							# print(tokens[j:j+k])
							ngrams[joined_token] = index
							index += 1

			else:
				ngrams[tokens[j]] = index
				index += 1

	return ngrams

if __name__ == '__main__':

	# Create ngrams
	tokens = ['__START__', 'I', 'am', 'happy', '.', '__END__']
	ngrams = convert_to_ngrams(tokens, 3, ' ')

	for ngram in ngrams:
		print(ngram)

