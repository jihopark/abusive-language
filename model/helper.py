#!/usr/bin/env python
""" Helper functions for training & evaluating the model"""

import warnings

import tensorflow as tf
from keras import backend as K
from sklearn import metrics

def calculate_metrics(y_true, y_pred, summary_writer=None, step=None,
        measureAccuracy=False, num_classes=2, classificationReport=False):
    # ignoring warning message
    # UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due
    # to no predicted samples.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        _average = "weighted" if num_classes > 2 else "binary"
        precision = metrics.precision_score(y_true, y_pred, average=_average)
        recall = metrics.recall_score(y_true, y_pred, average=_average)
        f1 = metrics.f1_score(y_true, y_pred, average=_average)
        if measureAccuracy:
            accuracy = metrics.accuracy_score(y_true, y_pred)
        else:
            accuracy = 0
        if classificationReport:
            if num_classes == 3:
                target_names = ["none", "racism", "sexism"]
            else:
                target_names = ["negative", "positive"]
            print(metrics.classification_report(y_true, y_pred, digits=3,
                                              target_names=target_names))

    if summary_writer != None and step != None:
        summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="precision",
                                                                      simple_value=precision)]), global_step=step)
        summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="recall",
                                                                      simple_value=recall)]), global_step=step)
        summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="f1",
                                                                      simple_value=f1)]), global_step=step)
        if measureAccuracy:
            summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="accuracy",
                                                                          simple_value=accuracy)]), global_step=step)

    return precision, recall, f1, accuracy


