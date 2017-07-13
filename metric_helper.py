from sklearn.metrics import classification_report
from keras.callbacks import Callback
import numpy as np


class ClassificationReport(Callback):
    def __init__(self, model, x_eval, y_eval, labels):
        self.model = model
        self.x_eval = x_eval
        self.truth = np.argmax(y_eval, axis=1)
        self.labels = labels

    def on_epoch_end(self, epoch, logs={}):
        print("Generating Classification Report:")
        preds = np.argmax(self.model.predict(self.x_eval, verbose=1), axis=1)
        print("\n%s\n" % classification_report(self.truth, preds, target_names=self.labels))
