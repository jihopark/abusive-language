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

from model.hybrid_cnn import HybridCNN

CONFIG_KEYS = [ # training parameters
               "batch_size", "num_epochs", "learning_rate",
               "include_davidson",
               # model parameters
               "word_filter_sizes", "char_filter_sizes",
               "num_filters",
               "use_embedding_layer", "train_embedding", "use_pretrain_embedding",
               # others
               "logdir", "model_name"]

parser =argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, required=True)
args = vars(parser.parse_args())

if args["config_file"]:
    FLAGS = config_helper.load(args["config_file"], CONFIG_KEYS)
    assert FLAGS["model_name"] == "hybrid_cnn"

print(FLAGS)

# create a summary writter for tensorboard visualization
log_folder = FLAGS["logdir"] if FLAGS["logdir"] else str(int(time.time()))
log_path = os.path.dirname(os.path.abspath(__file__)) +  "/logs/" + log_folder

# loading data

data_word, labels = data_helper.load_abusive_binary("word",
                                                FLAGS["include_davidson"],
                                                vectors=(not FLAGS["use_embedding_layer"]))

data_char, _ = data_helper.load_abusive_binary("char",
                                               FLAGS["include_davidson"])


with open("./data/word_outputs/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
    vocab_size = len(vocab["word2id"].keys())
    print("vocabulary loaded with %s words" % vocab_size)

embedding_matrix = None
if FLAGS["use_pretrain_embedding"]:
    embedding_matrix = np.load("./data/word_outputs/glove_embedding.npy")
    assert embedding_matrix.shape[0] == vocab_size
    assert embedding_matrix.shape[1] ==  200
    print("loaded pretrained embedding")

# defining model

K.set_learning_phase(1)

word_len = data_word["x_train"].shape[1]
char_len = data_char["x_train"].shape[1]
char_vocab = data_char["x_train"].shape[2]
print("word_len: %s, char_len: %s" % (word_len, char_len))

model = HybridCNN(word_len=word_len,
                  char_len=char_len,
                  n_classes=len(labels),
                  word_vocab_size=vocab_size,
                  char_vocab_size=char_vocab,
                  word_filter_sizes=list(map(int, FLAGS["word_filter_sizes"].split(","))),
                  char_filter_sizes=list(map(int, FLAGS["char_filter_sizes"].split(","))),
                  num_filters=FLAGS["num_filters"],
                  use_embedding_layer=FLAGS["use_embedding_layer"],
                  embedding_size=200,
                  embedding_matrix=embedding_matrix,
                  train_embedding=FLAGS["train_embedding"],
                  learning_rate=FLAGS["learning_rate"])

# define keras training procedure

tb_callback = TensorBoard(log_dir=log_path, histogram_freq=0,
                          write_graph=True, write_images=True)

clf_report_callback = ClassificationReport(model.model,
                                           [data_char["x_valid"], data_word["x_valid"]],
                                           data_word["y_valid"], labels)

ckpt_callback = ModelCheckpoint(log_path + "/weights.{epoch:02d}.hdf5",
                                monitor='val_acc', save_best_only=True,
                                save_weights_only=False, mode='max', verbose=1)

early_stop_callback = EarlyStopping(monitor='val_acc', min_delta=0, patience=1,
                                    verbose=0, mode='max')


model.model.fit(x=[data_char["x_train"], data_word["x_train"]],
                y=data_word["y_train"],
                batch_size=FLAGS["batch_size"],
                verbose=2, epochs=FLAGS["num_epochs"],
                callbacks=[tb_callback, clf_report_callback,
                          early_stop_callback, ckpt_callback],
               validation_data=([data_char["x_valid"], data_word["x_valid"]], data_word["y_valid"]))

print("Training Finished")



