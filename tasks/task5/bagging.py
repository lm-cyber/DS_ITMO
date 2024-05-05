import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelEncoder as le
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt

from typing import Type
import itertools

from dtree import Classifier, MyClassifier

class Bagger(Classifier):
    def __init__(self, Classifier_: Type[Classifier], n_estimators: int = 100, random_state:int=None, **classifier_params: dict):
        self.Classifier = Classifier_
        self.classifier_params = classifier_params
        self.n_estimators = n_estimators
        self.classifiers = []
        self.random_state = random_state
        self.n_classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.n_classes = np.unique(y).shape[0]
        np.random.seed(self.random_state)
        self.classifiers = [self.Classifier(**self.classifier_params) for i in range(self.n_estimators)]
        for clf in tqdm.tqdm(self.classifiers):
            sample = np.random.randint(low=0, high=X.shape[0], size=X.shape[0], dtype=int)
            n_features = math.ceil(X.shape[1] - np.sqrt(X.shape[1]))
            features = np.random.choice(np.arange(0, X.shape[1]), n_features, replace=True)
            sample_X = X[sample]
            sample_X[:, features] = 0
            sample_y = y[sample]
            sample_y = np.eye(self.n_classes)[sample_y]
            clf.fit(sample_X, sample_y, y_one_hot=True)


    def predict(self, X: np.ndarray) -> np.ndarray:
        predicts = np.zeros([X.shape[0], self.n_classes], dtype=float)
        out = np.zeros(X.shape[0], dtype=int)
        for i, clf in enumerate(self.classifiers):
            predicts += clf.predict_proba(X)
        print(predicts.shape)
        for i,pred in enumerate(predicts):
            out[i] = np.argmax(pred)
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y = np.zeros([X.shape[0], self.n_classes])
        for i, reg in enumerate(self.classifiers):
            y += reg.predict_proba(X)
        return y / len(self.classifiers)



def prepare():
    X = pd.read_csv('./X_train.csv')
    y = pd.read_csv('./y_train.csv')
    y['label'] = le().fit_transform(y.surface)

    FEATURE_COLUMNS = X.columns.to_list()[3:]

    X_norm = []
    y_norm = []
    for series_id, group in tqdm.tqdm(X.groupby("series_id")):
        features = group[FEATURE_COLUMNS]
        col_names = list(itertools.product(features.index, features.columns))
        X_norm.append([features.loc[comb] for comb in list(itertools.product(features.index, features.columns))])
        y_norm.append(y[y.series_id == series_id].iloc[0].label)

    X = pd.DataFrame(X_norm, columns=col_names)
    y = pd.Series(y_norm)

    df = pd.concat([X, y], axis=1)
    df = df.rename(columns={0: "y"})
    return df


