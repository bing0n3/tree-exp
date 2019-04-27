import numpy as np
from sklearn import metrics
import time as time


class Metric:
    def calculate_roc(self, x):
        pass

    def calculate_precision(self, y_true, y_pred):
        # multicalss problem need to be defined
        return metrics.precision_score(y_true, y_pred)

    def calculate_accuracy(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)

    def calculate_recall(self, y_true, y_pred):
        return metrics.recall_score(y_true, y_pred)
