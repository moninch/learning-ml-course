import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter


class MyKNNClf:
    def __init__(self, k=3, metric="euclidean"):
        self.k = k
        self.metric = metric
        self.train_size = None

    def __str__(self):
        return f"MyKNNClf class: k={self.k}"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X.values
        self.y = y.values
        self.train_size = X.shape

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _chebyshev_distance(self, x1, x2):
        return np.max(np.abs(x1 - x2))

    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def _cosine_distance(self, x1, x2):
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        return 1 - (dot_product / (norm_x1 * norm_x2))

    def predict(self, X: pd.DataFrame):
        return np.array([1 if prob >= 0.5 else 0 for prob in self.predict_proba(X)])

    def predict_proba(self, X: pd.DataFrame):
        # test = X.to_numpy()
        # distances = cdist(test, self.X, metric=self.metric)
        # indx = np.argsort(distances)[:, : self.k]
        # return np.mean(self.y[indx], axis=1)

        distances = self._compute_distances(X)
        indx = np.argsort(distances)[:, : self.k]
        return np.mean(self.y[indx], axis=1)

    def _compute_distances(self, X):
        test = X.to_numpy()
        distances = []
        for x_test in test:
            row_distances = []
            for x_train in self.X:
                if self.metric == "euclidean":
                    row_distances.append(self._euclidean_distance(x_test, x_train))
                elif self.metric == "chebyshev":
                    row_distances.append(self._chebyshev_distance(x_test, x_train))
                elif self.metric == "manhattan":
                    row_distances.append(self._manhattan_distance(x_test, x_train))
                elif self.metric == "cosine":
                    row_distances.append(self._cosine_distance(x_test, x_train))
            distances.append(row_distances)
        return np.array(distances)
