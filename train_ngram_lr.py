#!/usr/bin/env python
""" Training script for N-gram Linear Regression"""

import tensorflow as tf
import numpy as np

from data.ngrams import load_data_from_file
from data.preprocess import concat_unshared_task_datasets as load_data
from data.utils import split_dataset_binary, rand_batch_gen, batch_gen

from model.lr import LinearRegression

# Training parameters
tf.flags.DEFINE_integer("batch_size", 1, "Number of batch size (default: 1)")
tf.flags.DEFINE_integer("num_steps", 100000, "Number of training steps(default: 100000)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many epochs (default: 1000)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 1000)")

# Misc Parameters
tf.flags.DEFINE_integer("memory_usage_percentage", 80, "Set Memory usage percentage (default:80)")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
tf.logging.set_verbosity(tf.logging.INFO)


def train_batch(model, sess, train_batch_gen):
    # get batches
    batchX, batchY = train_batch_gen.__next__()

    # add x0 for bias
    batchX = np.hstack((np.ones((batchX.shape[0], 1)), batchX))
    feed_dict = {
        model.labels: batchY,
        model.X: batchX
        }
    sess.run([model.train_op], feed_dict)
    return feed_dict

def eval(model, sess, x_eval, y_eval):
    assert x_eval.shape[1] == model.n_dim - 1
    x_eval = np.hstack((np.ones((x_eval.shape[0], 1)), x_eval))
    feed_dict = {
        model.labels: y_eval,
        model.X: x_eval
        }
    return sess.run([model.cost, model.accuracy], feed_dict)

def train(model, train_set, valid_set, sess, train_iter):
    if not sess:
        return None
    sess.run(tf.global_variables_initializer())

    print("Training Started on " + model.name)
    for i in range(train_iter):
        try:
            feed_dict = train_batch(model, sess, train_set)
            if i % 100 == 0:
                cost, accuracy = sess.run([model.cost, model.accuracy], feed_dict)
                print("Iteration %s: mini-batch cost=%.4f, accuracy=%.3f" % (i, cost, accuracy))
            if i % FLAGS.evaluate_every == 0:
                valid_cost, valid_accuracy = eval(model, sess, x_valid, y_valid)
                print("\n**Validation set cost=%.4f, accuracy=%.3f" % (valid_cost, valid_accuracy))

        except KeyboardInterrupt:
            print('Interrupted by user at iteration{}'.format(i))
            return sess
    #TODO: evaluate test set

if __name__ == '__main__':
    name = "racism_binary_word_pad_none_3gram"
    # racism word ngram
    train_data, valid_data, test_data, metadata = load_data_from_file(name)

    x_train, y_train = train_data[:, :-1], train_data[:, [-1]]
    x_valid, y_valid = valid_data[:, :-1], valid_data[:, [-1]]
    x_test, y_test = test_data[:, :-1], test_data[:, [-1]]

    print("training data x_shape:%s, y_shape:%s" % (str(x_train.shape), str(y_train.shape)))
    print("validation data x_shape:%s, y_shape:%s" % (str(x_valid.shape), str(y_valid.shape)))
    print("test data x_shape:%s, y_shape:%s" % (str(x_test.shape), str(y_test.shape)))

    n_dim = x_train.shape[1] + 1 # incldue bias column
    print("\nInitializing the Logistic Regression Model with n_features=%s" % n_dim)
    model = LinearRegression(n_dim, name=name)

    # create session for training
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.memory_usage_percentage/100)
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=gpu_options)

    train_batch_generator = rand_batch_gen(x_train, y_train, FLAGS.batch_size)
    with tf.Session(config=session_conf) as sess:
        train(model, train_batch_generator, None, sess, FLAGS.num_steps)
