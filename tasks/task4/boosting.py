from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import RocCurveDisplay
from matplotlib import pyplot as plt

from typing import Type

from dtree import Classifier, MyClassifier

class GradientBooster(Classifier):
    def __init__(self,
                 Classifier_: Type[Classifier],
                 alpha=0.1,
                 n_estimators=100,
                 **Classifier_params: dict):
        self.Classifier = Classifier_
        self.Classifier_params = Classifier_params
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.regressors = []


    def fit(self, X: np.ndarray, y: np.ndarray, y_one_hot=False) -> None:
        if not y_one_hot:
            y = np.eye(np.unique(y).shape[0])[y]

        self.regressors.append(self.Classifier(**self.Classifier_params))
        self.regressors[0].fit(X, y, y_one_hot=True)
        out = self.regressors[0].predict_proba(X)
        for i in tqdm(range(1, self.n_estimators)):
            self.regressors.append(self.Classifier(**self.Classifier_params))
            softmax_out = np.exp(out) / np.expand_dims(np.sum(np.exp(out), axis=1), axis=1)
            grad = softmax_out - y
            self.regressors[i].fit(X, grad, y_one_hot=True)
            predicted_grad = self.regressors[i].predict_proba(X)
            out = out - self.alpha * predicted_grad


    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self.regressors[0].predict(X, y_one_hot=True)
        for i in range(1, self.n_estimators):
            out = out - self.alpha * self.regressors[i].predict(X, y_one_hot=True)
        return np.array([np.argmax(out[i]) for i in range(X.shape[0])])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = self.regressors[0].predict(X, y_one_hot=True)
        for i in range(1, self.n_estimators):
            out = out - self.alpha * self.regressors[i].predict(X, y_one_hot=True)
        softmax_out = np.exp(out) / np.expand_dims(np.sum(np.exp(out), axis=1), axis=1)
        return softmax_out


