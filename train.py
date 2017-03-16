#!/usr/bin/env python
""" Training script for N-gram Linear Regression"""
import warnings
import time
import os

import tensorflow as tf
from keras import backend as K
import numpy as np
from sklearn import metrics

from data.ngrams import load_data_from_file as load_data_ngram
from data.char import load_data_from_file as load_data_char
from data.utils import non_uniform_batch_gen, rand_batch_gen, batch_gen

from model.lr import LinearRegression
from model.char_cnn import CharCNN

# Training parameters
tf.flags.DEFINE_string("model_name", "char_cnn",
                       "Which model to train - char_cnn/ngram_lr (default=char_cnn")
tf.flags.DEFINE_integer("batch_size", 100, "Number of batch size (default: 100)")
tf.flags.DEFINE_integer("num_steps", 100000,
                        "Number of training steps(default: 100000)")
tf.flags.DEFINE_integer("evaluate_every", 1000,
                        "Evaluate model on dev set after this many epochs (default: 1000)")
tf.flags.DEFINE_integer("checkpoint_every", 1000,
                        "Save model after this many steps (default: 1000)")
tf.flags.DEFINE_float("learning_rate", 0.00001,
                      "Learning Rate of the model(default:0.00001)")
tf.flags.DEFINE_float("batch_pos_sample", 0.2,
                      "Percentage of positive samples in a batch (default:0.2)")


# CharCNN parameters
tf.flags.DEFINE_string("model_depth", "shallow",
                       "Depth of neural network model - choose shallow or deep (default:shallow)")
tf.flags.DEFINE_string("model_size", "small",
                       "Size of dimension of neural network model - choose \
                       small or large (default:small)")
tf.flags.DEFINE_integer("positive_weight", 5,
                        "Weight on the positive samples for calculating loss \
                        (default: 5)")


# Misc Parameters
tf.flags.DEFINE_integer("memory_usage_percentage", 90, "Set Memory usage percentage (default:90)")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
tf.logging.set_verbosity(tf.logging.INFO)


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
    return sess.run([model.merge_summary, model.cost, model.accuracy, model.prediction], feed_dict)

def calculate_metrics(y_true, y_pred, summary_writer, step):
    # ignoring warning message
    # UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due
    # to no predicted samples.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)

    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="precision",
                                                                  simple_value=precision)]), global_step=step)
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="recall",
                                                                  simple_value=recall)]), global_step=step)
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="f1",
                                                                  simple_value=f1)]), global_step=step)
    return precision, recall, f1

def train(model, train_set, valid_set, sess, train_iter):
    if not sess:
        return None
    sess.run(tf.global_variables_initializer())

    # create a summary writter for tensorboard visualization
    log_path = os.path.dirname(os.path.abspath(__file__)) +  "/logs/" + str(int(time.time()))

    train_writer = tf.summary.FileWriter(log_path + '/train', sess.graph)
    valid_writer = tf.summary.FileWriter(log_path + '/valid')

    print("Training Started with model: %s, log_path=%s" % (model.name,
                                                            log_path))
    for i in range(train_iter):
        try:
            feed_dict = train_batch(model, sess, train_set)
            if i % 100 == 0:
                #TODO: make training accuracy calculation for all 100 steps
                summary, cost, accuracy, pred = sess.run([model.merge_summary,
                                           model.cost,
                                           model.accuracy,
                                           model.prediction], feed_dict)
                train_precision, train_recall, train_f1 = calculate_metrics(feed_dict[model.labels], pred, train_writer, i)
                print("Iteration %s: mini-batch cost=%.4f, accuracy=%.3f" % (i, cost, accuracy))
                print("Precision=%.4f, Recall=%.4f, F1=%.4f" % (train_precision,
                                                                train_recall,
                                                                train_f1
                                                                ))
                train_writer.add_summary(summary, i)

                logits = sess.run(model.logits, feed_dict)
                print(logits[0:20])
            if i % FLAGS.evaluate_every == 0:
                summary, cost, accuracy, pred = eval(model, sess, x_valid, y_valid)
                valid_precision, valid_recall, valid_f1 = calculate_metrics(y_valid, pred, valid_writer, i)
                print("\n**Validation set cost=%.4f, accuracy=%.3f" % (cost,
                                                                       accuracy))
                print("Precision=%.4f, Recall=%.4f, F1=%.4f\n" % (valid_precision,
                                                                valid_recall,
                                                                valid_f1))
                valid_writer.add_summary(summary, i)
        except KeyboardInterrupt:
            print('Interrupted by user at iteration{}'.format(i))
            break
    train_writer.close()
    valid_writer.close()
    return sess

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
        # sexism binary
        name = "sexism_binary"

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
                        positive_weight=FLAGS.positive_weight)
    else:
        raise ValueError("Wrong model name. Please input from ngram_lr/char_cnn")

    print("training data x_shape:%s, y_shape:%s" % (str(x_train.shape), str(y_train.shape)))
    print("validation data x_shape:%s, y_shape:%s" % (str(x_valid.shape), str(y_valid.shape)))
    print("test data x_shape:%s, y_shape:%s" % (str(x_test.shape), str(y_test.shape)))

    # create session for training
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.memory_usage_percentage/100)
    session_conf = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    train_batch_generator = non_uniform_batch_gen(x_train,
                                                  y_train,
                                                  FLAGS.batch_size,
                                                  FLAGS.batch_pos_sample)
    with tf.Session(config=session_conf) as sess:
        K.set_session(sess)
        train(model, train_batch_generator, None, sess, FLAGS.num_steps)
        #TODO: session close and save
