import os
import re
import numpy as np
import pandas as pd



# minimum word count for a tweet. tweet less than this will be removed
MIN_WORDS = 2

def is_url(s):
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', s)
    return len(urls) > 0

def preprocess_tweet(tweet):
    try:
        if tweet[0:2] == "RT":
            tweet = tweet[2:]
        tokens = tweet.split()
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
    package_directory = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(package_directory + '/crawled/wassem_hovy_naacl.csv',
                     sep="\t",
                     header=None,
                     skiprows=[0],
                     names=["Tweet_ID", "Previous", "User_ID", "Text", "Label"],
                     error_bad_lines=False
                    )
    # wassem css 2016
    df2 = pd.read_csv(package_directory + '/crawled/wassem_css.csv',
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

if __name__ == '__main__':
    data = concat_unshared_task_datasets()

