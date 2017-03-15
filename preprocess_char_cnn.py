#!/usr/bin/env python
""" Preprocessing script for CharCNN"""

from data.char import preprocess_char_cnn
from data.preprocess import load_preprocessed_data

data = load_preprocessed_data("sexism_binary")

preprocess_char_cnn(data, "sexism_binary")

