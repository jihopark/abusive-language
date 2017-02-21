import re
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(reduce_len=True)

# minimum word count for a tweet. tweet less than this will be removed
MIN_WORDS = 2

def is_url(s):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', s)
    return len(urls) > 0

def preprocess_tweet(tweet):
    try:
        if tweet[0:2] == "RT":
            tweet = tweet[2:]
        tokens = tknzr.tokenize(tweet)
        tokens = filter(lambda x: x[0] != "@" and not is_url(x), tokens) # remove mentions or urls
        tokens = map(lambda x: x[1:] if x[0] == "#" else x, tokens) # remove # from hashtags
        tokens = [w.lower() for w in tokens] # to lower case
    except UnicodeDecodeError:
        print("unicode decode error")
        return ""
    except TypeError:
        print ("type error")
        return ""
    return " ".join(tokens) if len(tokens) >= MIN_WORDS else ""
