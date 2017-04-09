#!/usr/bin/env python
""" Training script for all the models"""
import time
import os

import tensorflow as tf
from keras import backend as K
import numpy as np

from data.word import load_data_from_file as load_data_cnn
from data.ngrams import load_data_from_file as load_data_ngram
from data.char import load_data_from_file as load_data_char
from data.hybrid import load_data_from_file as load_data_hybrid

from data.hybrid import extract_from_batch
from data.utils import balanced_batch_gen, rand_batch_gen, print_errors

from model.lr import LinearRegression
from model.char_cnn import CharCNN
from model.word_cnn import WordCNN
from model.hybrid_cnn import HybridCNN
from model.helper import calculate_metrics

# Training parameters
tf.flags.DEFINE_string("model_name", "word_cnn",
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

#WordCNN parameters
tf.flags.DEFINE_string("word_cnn_filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '1,2,3')")
tf.flags.DEFINE_integer("word_cnn_num_filters", 50, "Number of filters per filter size (default: 50)")


#Hybrid CNN parameters
tf.flags.DEFINE_string("hybrid_word_filter_sizes", "1,2,3",
                       "Comma-separated filter sizes (default: '1,2,3')")
tf.flags.DEFINE_string("hybrid_char_filter_sizes", "3,4,5",
                       "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("hybrid_cnn_num_filters", 50, "Number of filters per filter size (default: 50)")
tf.flags.DEFINE_integer("hybrid_cnn_pool_size", 1, "Number of filters per filter size (default: 1)")

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
    feed_dict = {model.labels: batchY}
    if FLAGS.model_name == "hybrid_cnn":
        batchW, batchC = extract_from_batch(batchX)
        feed_dict.update({model.X_word: batchW})
        feed_dict.update({model.X_char: batchC})
    else:
        feed_dict.update({model.X: batchX})
    feed_dict.update(add_drop_out(model, 0.5))
    sess.run([model.train_op], feed_dict)
    return feed_dict

def eval(model, sess, x_eval, y_eval):
    feed_dict = {model.labels: y_eval}

    if FLAGS.model_name == "hybrid_cnn":
        batchW, batchC = extract_from_batch(x_eval)
        feed_dict.update({model.X_word: batchW})
        feed_dict.update({model.X_char: batchC})
    else:
        feed_dict.update({model.X: x_eval})

    feed_dict.update(add_drop_out(model, 1.0))
    return sess.run([model.merge_summary, model.cost, model.prediction], feed_dict)

def add_drop_out(model, probability):
    if FLAGS.model_name == "word_cnn" or FLAGS.model_name == "hybrid_cnn":
        return {model.dropout_keep_prob: probability}
    elif FLAGS.model_name == "char_cnn":
        value = 1 if probability < 1 else 0
        return {K.learning_phase(): value}
    return {}

def save_ckpt(sess, saver, path):
    save_path = saver.save(sess, path)
    print("Model saved in file: %s" % save_path)

def error_analysis(x, true, pred, dictionary):
    if FLAGS.model_name == "hybrid_cnn":
        x, _ = extract_from_batch(x)
    print_errors(x, true, pred, FLAGS.model_name, dictionary)

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
                summary, cost, pred = eval(model, sess, valid_set["x"], valid_set["y"])
                valid_precision, valid_recall, valid_f1 = calculate_metrics(valid_set["y"],
                                                                            pred,
                                                                            valid_writer, i)
                error_analysis(valid_set["x"], valid_set["y"], pred,
                        model.dictionary if hasattr(model, "dictionary") else None)
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
    name = FLAGS.dataset_name
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
        (x_train, y_train,
         x_valid, y_valid,
         x_test, y_test,
         initW, vocab) = load_data_cnn(name)

        text_len = x_train.shape[1]
        vocab_size = len(vocab.vocabulary_)
        print("\nInitializing the WordCNN Model with vocab_size=%s,text_len=%s" \
                % (vocab_size, text_len))


        model = WordCNN(name, sequence_length=text_len,
                        n_classes=2,
                        vocab_size=vocab_size,
                        filter_sizes=list(map(int, FLAGS.word_cnn_filter_sizes.split(","))),
                        num_filters=FLAGS.word_cnn_num_filters,
                        embedding_size=300,
                        l2_reg_lambda=FLAGS.cnn_l2, embedding_static=True,
                        word2vec_multi=False,
                        learning_rate=FLAGS.learning_rate,
                        dictionary=vocab)
    elif FLAGS.model_name == "hybrid_cnn":
        (x_train, y_train,
         x_valid, y_valid,
         x_test, y_test,
         initW, vocab) = load_data_hybrid(name)

        word_text_len = x_train[0]["word"].shape[0]
        word_vocab_size = len(vocab.vocabulary_)

        char_text_len = x_train[0]["char"].shape[0]
        char_vocab_size = x_train[0]["char"].shape[1]

        model = HybridCNN(name,
                          word_len=word_text_len, char_len=char_text_len,
                          n_classes=2,
                          word_vocab_size=word_vocab_size,
                          char_vocab_size=char_vocab_size,
                          word_filter_sizes=list(map(int, FLAGS.hybrid_word_filter_sizes.split(","))),
                          char_filter_sizes=list(map(int, FLAGS.hybrid_char_filter_sizes.split(","))),
                          num_filters=FLAGS.hybrid_cnn_num_filters,
                          embedding_size=300,
                          pool_size=FLAGS.hybrid_cnn_pool_size,
                          l2_reg_lambda=FLAGS.cnn_l2, embedding_static=True,
                          learning_rate=FLAGS.learning_rate,
                          dictionary=vocab)
    else:
        raise ValueError("Wrong model name. Please input from \
                ngram_lr/char_cnn/word_cnn/hybrid_cnn")

    # create session for training
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.memory_usage_percentage/100)
    session_conf = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    train_batch_generator = balanced_batch_gen(x_train,
                                               y_train,
                                               FLAGS.batch_size)
    with tf.Session(config=session_conf) as sess:
        if FLAGS.model_name == "char_cnn":
            K.set_session(sess)
        elif FLAGS.model_name == "word_cnn":
            print("Initializing pre-trained word2vec embeddings \
                    shape init:%s tensor:%s"
                    % (str(initW.shape), str(model.W.get_shape())))
            sess.run(model.W.assign(initW))
        train(model, train_batch_generator, {"x": x_valid, "y": y_valid}, sess, FLAGS.num_steps)

