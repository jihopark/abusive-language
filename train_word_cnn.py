import os
import argparse
import time
import pickle

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import numpy as np
from keras import backend as K

import config_helper
from metric_helper import ClassificationReport
import data_helper

from model.word_cnn import WordCNN

CONFIG_KEYS = [ # training parameters
               "batch_size", "num_epochs", "learning_rate", 
               "include_davidson",
               # model parameters
               "filter_sizes", "num_filters",
               # others
               "logdir", "model_name"]

parser =argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, required=True)
args = vars(parser.parse_args())

if args["config_file"]:
    FLAGS = config_helper.load(args["config_file"], CONFIG_KEYS)
    assert FLAGS["model_name"] == "word_cnn"

print(FLAGS)

# create a summary writter for tensorboard visualization
log_folder = FLAGS["logdir"] if FLAGS["logdir"] else str(int(time.time()))
log_path = os.path.dirname(os.path.abspath(__file__)) +  "/logs/" + log_folder

# loading data

data, labels = data_helper.load_abusive_binary("word",
                                                FLAGS["include_davidson"])

with open("./data/word_outputs/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
    vocab_size = len(vocab["word2id"].keys())
    print("vocabulary loaded with %s words" % vocab_size)

# defining model

K.set_learning_phase(1)

sequence_length = data["x_train"].shape[1]
print("sequence_length: %s" % sequence_length)

model = WordCNN(sequence_length=sequence_length,
                n_classes=len(labels),
                vocab_size=vocab_size,
                filter_sizes=list(map(int, FLAGS["filter_sizes"].split(","))),
                num_filters=FLAGS["num_filters"],
                embedding_size=300,
                learning_rate=FLAGS["learning_rate"])

# define keras training procedure

tb_callback = TensorBoard(log_dir=log_path, histogram_freq=0,
                  write_graph=True, write_images=True)

clf_report_callback = ClassificationReport(model.model, data["x_valid"],
                                           data["y_valid"], labels)

ckpt_callback = ModelCheckpoint(log_path + "/weights.{epoch:02d}.hdf5",
                                monitor='val_acc', save_best_only=True,
                                save_weights_only=False, mode='max', verbose=1)

early_stop_callback = EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=0, mode='max')


model.model.fit(x=data["x_train"], y=data["y_train"],
               batch_size=FLAGS["batch_size"],
               verbose=2, epochs=FLAGS["num_epochs"],
               callbacks=[tb_callback, clf_report_callback,
                          early_stop_callback, ckpt_callback],
               validation_data=(data["x_valid"], data["y_valid"]))

print("Training Finished")



