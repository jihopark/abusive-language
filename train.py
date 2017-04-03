#!/usr/bin/env python
""" Training script for all the models"""
import time
import os

import tensorflow as tf
from keras import backend as K
import numpy as np
import gensim

from data.word import load_data_from_file as load_data_cnn
from data.ngrams import load_data_from_file as load_data_ngram
from data.char import load_data_from_file as load_data_char
from data.char import print_errors as print_errors_char
from data.utils import balanced_batch_gen, rand_batch_gen

from model.lr import LinearRegression
from model.char_cnn import CharCNN
from model.word_cnn import WordCNN
from model.helper import calculate_metrics

# Training parameters
tf.flags.DEFINE_string("model_name", "char_cnn",
                       "Which model to train - char_cnn/ngram_lr (default=char_cnn")
tf.flags.DEFINE_integer("batch_size", 32, "Number of batch size (default: 32)")
tf.flags.DEFINE_integer("num_steps", 400000,
                        "Number of training steps(default: 400000)")
tf.flags.DEFINE_integer("evaluate_every", 5000,
                        "Evaluate model on dev set after this many epochs \
                        (default: 5000)")
tf.flags.DEFINE_integer("checkpoint_every", 50000,
                        "Save model after this many steps (default: 50000)")
tf.flags.DEFINE_float("learning_rate", 0.001,
                      "Learning Rate of the model(default:0.001")
tf.flags.DEFINE_string("dataset_name", "sexism_binary",
                       "Which dataset to train (default=sexism_binary")


# CharCNN parameters
tf.flags.DEFINE_string("model_depth", "shallow",
                       "Depth of neural network model - choose shallow,deep\
                       (default:shallow)")
tf.flags.DEFINE_string("model_size", "large",
                       "Size of dimension of neural network model - choose \
                       small or large (default:large)")
tf.flags.DEFINE_integer("positive_weight", 1,
                        "Weight on the positive samples for calculating loss \
                        (default: 1)")
tf.flags.DEFINE_integer("filter_size", 4,
                        "Filter size of the CNN kernel \
                        (default: 4 - will use default setting)")
tf.flags.DEFINE_string("max_pool_type", "normal_6",
                        "How to configure the max_pool \
                         normal_6=pool_size is all 6, \
                         normal_3=pool_size is all 3, \
                         half_6=pool_size is 6 and 3 \
                         (default: normal_6")
tf.flags.DEFINE_float("cnn_l1", 10,
                        "L1 regularizer weight on CNN layers \
                        (default: 10)")
tf.flags.DEFINE_float("fully_connected_l1", 0,
                        "L1 regularizer weight on fully connected layers \
                        (default: 0)")
tf.flags.DEFINE_float("cnn_l2", 1,
                        "L2 regularizer weight on CNN layers \
                        (default: 1)")
tf.flags.DEFINE_float("fully_connected_l2", 0,
                        "L2 regularizer weight on fully connected layers \
                        (default: 0)")

# Misc Parameters
tf.flags.DEFINE_integer("memory_usage_percentage", 90, "Set Memory usage percentage (default:90)")
tf.flags.DEFINE_string("log_dir", "",
                       "Where to save the log files (default: timestamp)")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
tf.logging.set_verbosity(tf.logging.ERROR)


def train_batch(model, sess, train_batch_gen):
    # get batches
    batchX, batchY = train_batch_gen.__next__()
    feed_dict = {
        model.labels: batchY,
        model.X: batchX,
        K.learning_phase(): 1 # whether to use dropout or not
        }
    sess.run([model.train_op], feed_dict)
    return feed_dict

def eval(model, sess, x_eval, y_eval):
   # assert x_eval.shape[1] == model.n_dim
    feed_dict = {
        model.labels: y_eval,
        model.X: x_eval,
        K.learning_phase(): 0 # whether to use dropout or not
        }
    return sess.run([model.merge_summary, model.cost, model.prediction], feed_dict)

def save_ckpt(sess, saver, path):
    save_path = saver.save(sess, path)
    print("Model saved in file: %s" % save_path)

def print_errors(x, true, pred):
    if FLAGS.model_name == "char_cnn":
        print_errors_char(x, true, pred)

def train(model, train_set, valid_set, sess, train_iter):
    if not sess:
        return None
    sess.run(tf.global_variables_initializer())

    # create a summary writter for tensorboard visualization
    log_folder = FLAGS.log_dir if FLAGS.log_dir else str(int(time.time()))
    log_path = os.path.dirname(os.path.abspath(__file__)) +  "/logs/" + log_folder
    train_writer = tf.summary.FileWriter(log_path + '/train', sess.graph)
    valid_writer = tf.summary.FileWriter(log_path + '/valid')

    # create a saver object to save and restore variables
    # https://www.tensorflow.org/programmers_guide/variables#checkpoint_files
    saver = tf.train.Saver()
    os.makedirs(log_path + "/ckpt")
    ckpt_path =  log_path + "/ckpt"

    print("Training Started with model: %s, log_path=%s" % (model.name,
                                                            log_path))
    for i in range(train_iter):
        try:
            feed_dict = train_batch(model, sess, train_set)
            if i % 100 == 0:
                summary, cost, pred = sess.run([model.merge_summary,
                                           model.cost,
                                           model.prediction], feed_dict)
                train_precision, train_recall, train_f1 = calculate_metrics(feed_dict[model.labels], pred, train_writer, i)
                print("Iteration %s: mini-batch cost=%.4f" % (i, cost))
                print("Precision=%.4f, Recall=%.4f, F1=%.4f" % (train_precision,
                                                                train_recall,
                                                                train_f1
                                                                ))
                train_writer.add_summary(summary, i)

                logits = sess.run(model.logits, feed_dict)
                print(logits[0:20])
            if i % FLAGS.evaluate_every == 0:
                summary, cost, pred = eval(model, sess, x_valid, y_valid)
                valid_precision, valid_recall, valid_f1 = calculate_metrics(y_valid, pred, valid_writer, i)
                print_errors(x_valid, y_valid, pred)
                print("\n**Validation set cost=%.4f" % cost)
                print("Precision=%.4f, Recall=%.4f, F1=%.4f\n" % (valid_precision,
                                                                valid_recall,
                                                                valid_f1))
                valid_writer.add_summary(summary, i)
            if i % FLAGS.checkpoint_every == 0:
                save_ckpt(sess, saver, ckpt_path + ("/model-%s.ckpt" % i))
        except KeyboardInterrupt:
            print('Interrupted by user at iteration{}'.format(i))
            break
    K.set_learning_phase(0)
    save_ckpt(sess, saver, ckpt_path + "/model-final.ckpt")
    train_writer.close()
    valid_writer.close()
    sess.close()

if __name__ == '__main__':
    if FLAGS.model_name == "ngram_lr":
        # racism word ngram
        name = "racism_binary_word_pad_none_3gram"
        train_data, valid_data, test_data, metadata = load_data_ngram(name)

        x_train, y_train = train_data[:, :-1], train_data[:, [-1]]
        x_valid, y_valid = valid_data[:, :-1], valid_data[:, [-1]]
        x_test, y_test = test_data[:, :-1], test_data[:, [-1]]

        # add x0 for bias
        x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
        x_valid = np.hstack((np.ones((x_valid.shape[0], 1)), x_valid))
        x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

        n_dim = x_train.shape[1]
        print("\nInitializing the Logistic Regression Model with n_features=%s" % n_dim)
        model = LinearRegression(n_dim, name=name)
    elif FLAGS.model_name == "char_cnn":
        name = FLAGS.dataset_name

        x_train, y_train, x_valid, y_valid, x_test, y_test = load_data_char(name)
        text_len = x_train.shape[1]
        vocab_size = x_train.shape[2]

        print("\nInitializing the CharCNN Model with vocab_size=%s,text_len=%s" \
        % (vocab_size, text_len))

        model = CharCNN(name,
                        text_len=text_len,
                        vocab_size=vocab_size,
                        n_classes=2,
                        model_size=FLAGS.model_size,
                        model_depth=FLAGS.model_depth,
                        learning_rate=FLAGS.learning_rate,
                        positive_weight=FLAGS.positive_weight,
                        max_pool_type=FLAGS.max_pool_type,
                        kernel_size=FLAGS.filter_size,
                        cnn_l1=FLAGS.cnn_l1,
                        cnn_l2=FLAGS.cnn_l2,
                        fully_connected_l1=FLAGS.fully_connected_l1,
                        fully_connected_l2=FLAGS.fully_connected_l2)
    elif FLAGS.model_name == "word_cnn":
        x_train, y_train, x_valid, y_valid, x_test, y_test, initW, vocab = load_data_cnn(FLAGS.dataset_name)
    else:
        raise ValueError("Wrong model name. Please input from ngram_lr/char_cnn")

    print("training data x_shape:%s, y_shape:%s" % (str(x_train.shape), str(y_train.shape)))
    print("validation data x_shape:%s, y_shape:%s" % (str(x_valid.shape), str(y_valid.shape)))
    print("test data x_shape:%s, y_shape:%s" % (str(x_test.shape), str(y_test.shape)))

    # create session for training
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.memory_usage_percentage/100)
    session_conf = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    train_batch_generator = balanced_batch_gen(x_train,
                                               y_train,
                                               FLAGS.batch_size)
    with tf.Session(config=session_conf) as sess:
        K.set_session(sess)
        train(model, train_batch_generator, None, sess, FLAGS.num_steps)
