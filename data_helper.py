import random
import numpy as np
from keras.utils import to_categorical

splits = ["train", "valid", "test"]

def convert_id_to_vectors(ids, embedding_matrix):
    splits = ["x_train", "x_valid", "x_test"]
    for split in splits:
        x = np.zeros((len(ids[split]), ids[split].shape[1], 200))
        for i, row in enumerate(ids[split]):
            m = np.zeros((len(row), 200))
            for j, word in enumerate(row):
                m[j] = embedding_matrix[word]
            x[i] = m
        ids[split] = x
        print("%s turned into %s" % (split, str(ids[split].shape)))
    return ids

def load_waasem(path):
    data_w = {}
    for split in splits:
        data_w[split] = {}
        for label in ["none", "sexism", "racism"]:
            data_w[split][label] = np.load(path + "%s_%s_%s.npy"
                                           % (split, label, "waasem"))
    return data_w

def make_into_categorical(original_data, labels):
    data = {}
    for split in original_data.keys():
        x = "x_" + split
        y = "y_" + split
        data[x] = None
        data[y] = []
        for i, label in enumerate(labels):
            _data = original_data[split][label]
            if data[x] is not None:
                data[x] = np.vstack((data[x], _data))
            else:
                data[x] = _data
            print("split:%s, label:%s, data shape:%s" %
                  (split, label, str(data[x].shape)))
            data[y] += [i for _ in range(len(_data))]
        data[y] = to_categorical(data[y], num_classes=len(labels))

    return data

def load_abusive_binary(_type, include_davidson=True, include_relabel=True,
                        vectors=False):

    if _type in ["char", "word"]:
        path = "./data/%s_outputs/" % _type
    else:
        raise ValueError("wrong type")

    data_w = load_waasem(path)

    if include_davidson:
        data_d = {}
        for split in splits:
            data_d[split] = {}
            for label in ["none", "abusive"]:
                data_d[split][label] = np.load(path + "%s_%s_%s.npy"
                                               % (split, label, "davidson"))

        # merge
        for split in splits:
            for label in ["none", "sexism", "racism"]:
                if label == "none":
                    data_d[split]["none"] = np.vstack((data_d[split]["none"],
                                                       data_w[split]["none"]))
                else:
                    data_d[split]["abusive"] = np.vstack((data_d[split]["abusive"],
                                                          data_w[split][label]))
        data_w = data_d
    else:
        for split in splits:
            data_w[split]["abusive"] = np.vstack((data_w[split]["sexism"], data_w[split]["racism"]))

    if include_relabel:
        data_relabel = np.load(path + "train_none_relabel.npy")
        data_w["train"]["none"] = np.vstack((data_w["train"]["none"], data_relabel))
        print("added relabel to training set")

    labels = ["none", "abusive"]
    data = make_into_categorical(data_w, labels)

    if vectors:
        embedding_matrix = np.load("./data/word_outputs/glove_embedding.npy")
        data = convert_id_to_vectors(data, embedding_matrix)
    return data, labels

def load_mixed_testset(_type):
    if _type in ["char", "word"]:
        path = "./data/%s_outputs/" % _type
    else:
        raise ValueError("wrong type")

    data_w = load_waasem(path)

    data_d = {}
    data_d["test"] = {}
    for label in ["none", "abusive"]:
        data_d["test"][label] = np.load(path + "%s_%s_%s.npy"
                                       % ("test", label, "davidson"))
    sample_none_n = len(data_w["test"]["none"])
    sample_abusive_n = len(data_w["test"]["sexism"]) + len(data_w["test"]["racism"])

    print("sampling %s none and %s abusive label from davidson" %
            (sample_none_n, sample_abusive_n))

    none_idx = np.random.randint(len(data_d["test"]["none"]), size=sample_none_n)
    abusive_idx = np.random.randint(len(data_d["test"]["abusive"]),
            size=sample_abusive_n)


    data_d["test"]["none"] = data_d["test"]["none"][none_idx, :]
    print(data_d["test"]["none"].shape)

    data_d["test"]["abusive"] = data_d["test"]["abusive"][abusive_idx, :]
    print(data_d["test"]["abusive"].shape)

    for label in ["none", "sexism", "racism"]:
        if label == "none":
            data_d["test"]["none"] = np.vstack((data_d["test"]["none"],
                                        data_w["test"]["none"]))
        else:
            data_d["test"]["abusive"] = np.vstack((data_d["test"]["abusive"],
                                           data_w["test"][label]))

    labels = ["none", "abusive"]
    return make_into_categorical(data_d, labels), labels

def load_multiclass(_type, include_davidson=True):
    if _type in ["char", "word"]:
        path = "./data/%s_outputs/" % _type
    else:
        raise ValueError("wrong type")

    data_w = load_waasem(path)

    if include_davidson:
        data_d = {}
        for split in splits:
            data_d[split] = {}
            for label in ["none", "abusive"]:
                data_d[split][label] = np.load(path + "%s_%s_%s.npy"
                                               % (split, label, "davidson"))

        # merge
        for split in splits:
            for label in ["none", "sexism", "racism"]:
                if label == "none":
                    data_w[split]["none"] = np.vstack((data_w[split]["none"],
                                                       data_d[split]["none"]))

    labels = data_w["train"].keys()
    return make_into_categorical(data_w, labels), labels


def load_sexism_racism_binary(_type):
    if _type in ["char", "word"]:
        path = "./data/%s_outputs/" % _type
    else:
        raise ValueError("wrong type")

    data_w = load_waasem(path)
    labels = ["sexism", "racism"]

    return make_into_categorical(data_w, labels), labels





