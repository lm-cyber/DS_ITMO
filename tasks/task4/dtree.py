import numpy as np
from sklearn.tree import DecisionTreeRegressor

from dataclasses import dataclass
import sys

class Classifier:
    def __init__(self, *args, **kwargs):
        pass
    def fit(self, X: np.ndarray, y: np.ndarray, y_one_hot: bool=False) -> None:
        pass
    def predict(self, X: np.ndarray, y_one_hot: bool=False) -> np.ndarray:
        pass
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

class Regressor:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

class DTreeRegressor(Regressor):
    def __init__(self, max_depth: int=100):
        self.max_depth = max_depth
        self.fork_layers = []

    @dataclass
    class __Fork:
        leaf_index: int
        feature: int
        value: float
        left_mean: float
        right_mean: float

    @staticmethod
    def __get_inc_dec_variance(new_el, left_E_squared, left_E_of_square, right_E_squared, right_E_of_square, j, length):
        left_E_squared = ((np.sqrt(left_E_squared) * j + new_el) / (j + 1)) ** 2
        left_E_of_square = (left_E_of_square * j + new_el ** 2) / (j + 1)
        right_E_squared = ((np.sqrt(right_E_squared) * (length - j) - new_el) / (
                    length - j - 1)) ** 2
        right_E_of_square = (right_E_of_square * (length - j) - new_el ** 2) / (
                    length - j - 1)
        left_variance = left_E_of_square - left_E_squared
        right_variance = right_E_of_square - right_E_squared
        variance_x_len = left_variance * j + right_variance * (length - j)

        return left_E_squared, left_E_of_square, right_E_squared, right_E_of_square, variance_x_len


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        leaf_indexes = np.ones(X.shape[0], dtype=int)
        self.fork_layers = []
        for depth in range(self.max_depth):
            self.fork_layers.append([])
            for leaf_index in np.unique(leaf_indexes):
                min_variance_len = sys.float_info.max
                leaf_X = X[leaf_indexes == leaf_index]
                leaf_y = y[leaf_indexes == leaf_index]
                for feature in range(X.shape[1]):
                    sorted_leaf_X, sorted_leaf_y = np.array([*zip(*sorted(zip(leaf_X[:,feature], leaf_y), key=lambda el: el[0]))])
                    left_E_squared = sorted_leaf_y[0] ** 2
                    left_E_of_square = sorted_leaf_y[0] ** 2
                    right_E_squared = np.mean(sorted_leaf_y[1:]) ** 2
                    right_E_of_square = np.mean(sorted_leaf_y[1:] ** 2)
                    left_variance = left_E_of_square - left_E_squared
                    right_variance = right_E_of_square - right_E_squared
                    variance_x_len = left_variance + right_variance * (sorted_leaf_X.shape[0] - 1)
                    for j in range(1, sorted_leaf_X.shape[0] - 1):
                        if variance_x_len < min_variance_len:
                            min_variance_len = variance_x_len
                            dividing_feature = feature
                            dividing_value = sorted_leaf_X[j]
                            left_mean = np.mean(sorted_leaf_y[:j])
                            right_mean = np.mean(sorted_leaf_y[j:])
                        left_E_squared, left_E_of_square, right_E_squared, right_E_of_square, variance_x_len = \
                            DTreeRegressor.__get_inc_dec_variance(sorted_leaf_y[j],
                                                                  left_E_squared,
                                                                  left_E_of_square,
                                                                  right_E_squared,
                                                                  right_E_of_square,
                                                                  j,
                                                                  sorted_leaf_X.shape[0])

                if min_variance_len < sys.float_info.max:
                    self.fork_layers[depth].append(
                        self.__Fork(leaf_index, dividing_feature, dividing_value, left_mean, right_mean)
                    )
                    leaf_indexes[leaf_indexes == leaf_index] *= 2
                    condition = np.arange(0, leaf_indexes.shape[0])[leaf_indexes == leaf_index * 2]\
                        [leaf_X[:,dividing_feature] >= dividing_value]
                    leaf_indexes[condition] += 1


    def predict(self, X: np.ndarray) -> np.ndarray:
        y = np.zeros(X.shape[0], dtype=float)
        leaf_indexes = np.ones(X.shape[0], dtype=int)
        for forks in self.fork_layers:
            for fork in forks:
                X_leaf = X[leaf_indexes == fork.leaf_index]
                left_condition = np.arange(0, leaf_indexes.shape[0])[leaf_indexes == fork.leaf_index]\
                    [X_leaf[:,fork.feature] < fork.value]
                right_condition = np.arange(0, leaf_indexes.shape[0])[leaf_indexes == fork.leaf_index]\
                    [X_leaf[:,fork.feature] >= fork.value]
                y[left_condition] = fork.left_mean
                y[right_condition] = fork.right_mean
                leaf_indexes[leaf_indexes == fork.leaf_index] *= 2
                condition = np.arange(0, leaf_indexes.shape[0])[leaf_indexes == fork.leaf_index * 2]\
                    [X_leaf[:,fork.feature] >= fork.value]
                leaf_indexes[condition] += 1
        return y

class DTreeClassifier(Classifier):
    def __init__(self, max_depth=100):
        self.max_depth = max_depth
        self.regressors = []

    def fit(self, X: np.ndarray, y: np.ndarray, y_one_hot=False) -> None:
        if not y_one_hot:
            y = np.eye(np.unique(y).shape[0])[y]
        self.regressors = []
        for i in range(y.shape[1]):
            self.regressors.append(DTreeRegressor(max_depth=self.max_depth))
            self.regressors[i].fit(X, y[:,i])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y = np.zeros([X.shape[0], len(self.regressors)])
        for i, reg in enumerate(self.regressors):
            y[:, i] = reg.predict(X)

        return y

    def predict(self, X: np.ndarray, y_one_hot=False) -> np.ndarray:
        out = np.zeros(X.shape[0])
        y = self.predict_proba(X)
        if y_one_hot:
            for i,_ in enumerate(y):
                max_ = np.argmax(y[i])
                y[i] = 0
                y[i, max_] = 1
            return y

        for i,_ in enumerate(y):
            out[i] = np.argmax(y[i])
        return out

class MyClassifier(Classifier):
    def __init__(self, max_depth):
        self.regressors = []
        self.max_depth=max_depth
    def fit(self, X: np.ndarray, y: np.ndarray, y_one_hot: bool) -> None:
        if not y_one_hot:
            y = np.eye(np.unique(y).shape[0])[y]
        self.regressors = [DecisionTreeRegressor(max_depth=3) for i in range(y.shape[1])]
        for i in range(y.shape[1]):
            self.regressors[i].fit(X, y[:, i])

    def predict(self, X: np.ndarray, y_one_hot: bool) -> np.ndarray:
        out = np.zeros(X.shape[0])
        y = self.predict_proba(X)
        if y_one_hot:
            for i,_ in enumerate(y):
                max_ = np.argmax(y[i])
                y[i] = 0
                y[i, max_] = 1
            return y

        for i,_ in enumerate(y):
            out[i] = np.argmax(y[i])
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y = np.zeros([X.shape[0], len(self.regressors)])
        for i, reg in enumerate(self.regressors):
            y[:, i] = reg.predict(X)

        return y