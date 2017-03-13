#!/usr/bin/env python
""" Preprocessing script for N-gram Linear Regression"""

from data.ngrams import make_ngram_matrix
from data.preprocess import load_preprocessed_data

data = load_preprocessed_data("sexism_binary")

# racism word ngram

make_ngram_matrix(data["x_train"], data["y_train"],
                  data["x_valid"], data["y_valid"],
                  data["x_test"], data["y_test"],
                  n=3,
                  data_name="racism_binary_word_pad_none_3gram",
                  show_tqdm=True)
"""
make_ngram_matrix([data["racism"], data["none"]], n=4,
                  data_name="racism_binary_word_pad_1_4gram")
make_ngram_matrix([data["racism"], data["none"]], n=3,
                  data_name="racism_binary_word_pad_none_3gram_remove_punc",
                  tokenizer_options={"includePunct":False})
make_ngram_matrix([data["racism"], data["none"]], n=4,
                  data_name="racism_binary_word_pad_1_4gram_remove_punc",
                  tokenizer_options={"includePunct":False})
"""
# sexism word ngram
"""
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
"""
# racism char ngram
"""
make_ngram_matrix([data["racism"], data["none"]], n=4, m=3,
                  word_or_char="char",
                  data_name="racism_binary_char_pad_1_4gram")
make_ngram_matrix([data["racism"], data["none"]], n=4, m=3,
                  word_or_char="char",
                  data_name="racism_binary_char_pad_1_4gram_remove_punc",
                  tokenizer_options={"includePunct":False})
make_ngram_matrix([data["racism"], data["none"]], n=5, m=3,
                  word_or_char="char",
                  data_name="racism_binary_char_pad_1_5gram")
make_ngram_matrix([data["racism"], data["none"]], n=5, m=3,
                  word_or_char="char",
                  data_name="racism_binary_char_pad_1_5gram_remove_punc",
                  tokenizer_options={"includePunct":False})
"""
# sexism char ngram
"""
make_ngram_matrix([data["sexism"], data["none"]], n=4, m=3,
                  word_or_char="char",
                  data_name="sexism_binary_char_pad_1_4gram")
make_ngram_matrix([data["sexism"], data["none"]], n=4, m=3,
                  word_or_char="char",
                  data_name="sexism_binary_char_pad_1_4gram_remove_punc",
                  tokenizer_options={"includePunct":False})
make_ngram_matrix([data["sexism"], data["none"]], n=5, m=3,
                  word_or_char="char",
                  data_name="sexism_binary_char_pad_1_5gram")
make_ngram_matrix([data["sexism"], data["none"]], n=5, m=3,
                  word_or_char="char",
                  data_name="sexism_binary_char_pad_1_5gram_remove_punc",
                  tokenizer_options={"includePunct":False})
"""
