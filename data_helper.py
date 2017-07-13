import numpy as np
from keras.utils import to_categorical

splits = ["train", "valid", "test"]

def load_waasem(path):
    data_w = {}
    for split in splits:
        data_w[split] = {}
        for label in ["none", "sexism", "racism"]:
            data_w[split][label] = np.load(path + "%s_%s_%s.npy"
                                           % (split, label, "waasem"))
    return data_w

def make_into_categorical_binary(original_data, labels):
    data = {}
    for split in splits:
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
        data[y] = to_categorical(data[y], num_classes=2)

    return data

def load_abusive_binary(_type, include_davidson=True):

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

    labels = ["none", "abusive"]
    return make_into_categorical_binary(data_w, labels), labels

def load_sexism_racism_binary(_type):
    if _type in ["char", "word"]:
        path = "./data/%s_outputs/" % _type
    else:
        raise ValueError("wrong type")

    data_w = load_waasem(path)
    labels = ["sexism", "racism"]

    return make_into_categorical_binary(data_w, labels), labels





