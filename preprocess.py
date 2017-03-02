from sklearn.model_selection import train_test_split
import pandas as pd
import re
import numpy as np
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

def dataframe_to_list(df):
    df = df.drop_duplicates()
    result = list(map(lambda x: preprocess_tweet(x), df.tolist()))
    return list(filter(lambda x: x, result)) # filter out empty string

def concat_unshared_task_datasets():
    data = {}
    # waseem and hovy naacl 2016
    df = pd.read_csv('./crawled/wassem_hovy_naacl.csv',
                        sep="\t",
                        header=None,
                        skiprows=[0],
                        names=["Tweet_ID", "Previous", "User_ID", "Text", "Label"],
                        error_bad_lines=False
                    )
    # wassem css 2016
    df2 = pd.read_csv('./crawled/wassem_css.csv',
                        sep="\t",
                        header=None,
                        skiprows=[0],
                        names=["Tweet_ID", "Previous",
                        "User_ID", "Text",
                        "Expert_Label",
                        "Amateur_Label"],
                    error_bad_lines=False)
    # label: sexism
    data["sexism"] = dataframe_to_list(pd.concat([df[df['Label'] == 'sexism']['Text'],
                                          df2[df2['Expert_Label'] == 'sexism']['Text'],
                                          df2[df2['Expert_Label'] == 'both']['Text']]))
    # label: racism
    data["racism"] = dataframe_to_list(pd.concat([df[df['Label'] == 'racism']['Text'],
                                          df2[df2['Expert_Label'] == 'racism']['Text'],
                                          df2[df2['Expert_Label'] == 'both']['Text']]))
    # label: none
    data["none"] = dataframe_to_list(pd.concat([df[df['Label'] == 'none']['Text'],
                                          df2[df2['Expert_Label'] == 'neither']['Text']]))
    print("Unshared task dataset concat done.")
    print("Label Count: Sexism-%s, Racism-%s, None-%s" % (len(data["sexism"]),
                                                           len(data["racism"]),
                                                            len(data["none"])))
    return data

#TODO: create multi-class dataset split function

def create_binary_dataset(x_neg, x_pos, split=[0.7, 0.15, 0.15]):
	assert (split[0] + split[1] + split[2]) == 1

	x_neg = np.array(x_neg)
	x_pos = np.array(x_pos)

	y_neg = np.zeros_like(x_neg)
	y_pos = np.ones_like(x_pos)

	# create training set
	x_neg_train, x_neg_test, y_neg_train, y_neg_test = train_test_split(x_neg, y_neg, test_size=split[1]+split[2])
	x_pos_train, x_pos_test, y_pos_train, y_pos_test = train_test_split(x_pos, y_pos, test_size=split[1]+split[2])

	x_train = np.concatenate([x_neg_train, x_pos_train])
	y_train = np.concatenate([y_neg_train, y_pos_train])

	print("Training set size:%s (neg:%s/pos:%s)\n" % (len(x_train), len(x_neg_train), len(x_pos_train)))
	print("sample neg - " + x_train[0])
	print("sample pos - " + x_train[-1])

	# creating validation/test set
	test_split = split[2]/(split[1]+split[2])
	x_neg_valid, x_neg_test, y_neg_valid, y_neg_test = train_test_split(x_neg_test, y_neg_test, test_size=test_split)
	x_pos_valid, x_pos_test, y_pos_valid, y_pos_test = train_test_split(x_pos_test, y_pos_test, test_size=test_split)

	x_valid = np.concatenate([x_neg_valid, x_pos_valid])
	y_valid = np.concatenate([y_neg_valid, y_pos_valid])

	print("\nValidation set size:%s (neg:%s/pos:%s)\n" % (len(x_valid), len(x_neg_valid), len(x_pos_valid)))
	print("sample neg - " + x_valid[0])
	print("sample pos - " + x_valid[-1])

	x_test = np.concatenate([x_neg_test, x_pos_test])
	y_test = np.concatenate([y_neg_test, y_pos_test])

	print("\nTest set size:%s (neg:%s/pos:%s)\n" % (len(x_test), len(x_neg_test), len(x_pos_test)))
	print("sample neg - " + x_test[0])
	print("sample pos - " + x_test[-1])

	return x_train, y_train, x_valid, y_valid, x_test, y_test

if __name__ == '__main__':
    data = concat_unshared_task_datasets()
    create_binary_dataset(data["none"], data["racism"])
    create_binary_dataset(data["none"], data["sexism"],split=[0.8, 0.1, 0.1])
