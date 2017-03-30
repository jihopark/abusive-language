#!/usr/bin/env python
"""Functions related to handling sentiment twitter dataset"""

import os
from . import preprocess

def load_tweets():
    file_names = ["training_neg.csv", "training_pos.csv"]
    path = os.path.dirname(os.path.abspath(__file__)) + "/sentiment/"
    tweets = []
    if not os.path.isfile(path + "tweets_preprocessed.txt"):
        for name in file_names:
            with open(path + name, "r", encoding='utf-8', errors='ignore') as f:
                for line in f:
                    tweets.append(preprocess.preprocess_tweet(line.rstrip()))
        with open(path + "tweets_preprocessed.txt", "w") as f:
            for t in tweets:
                f.writelines(t + "\n")
        print("preprocessed and loaded %s" % len(tweets))
    else:
        with open(path + "tweets_preprocessed.txt", "r") as f:
            for line in f:
                tweets.append(line.rstrip())
        print("loaded %s" % len(tweets))

    return tweets
