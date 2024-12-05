import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter


class MyKNNClf:
    def __init__(self, k=3):
        self.k = k
        self.train_size = None

    def __str__(self):
        return f"MyKNNClf class: k={self.k}"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X.values
        self.y = y.values
        self.train_size = X.shape

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X: pd.DataFrame):
        return np.array([1 if prob >= 0.5 else 0 for prob in self.predict_proba(X)])

    def predict_proba(self, X: pd.DataFrame):
        train = np.expand_dims(self.X, axis=0)
        test = np.expand_dims(X.to_numpy(), axis=1)
        distances = np.sqrt(np.sum((test - train) ** 2, axis=-1))
        indx = np.argsort(distances)[:, : self.k]

        return np.mean(self.y[indx], axis=1)
