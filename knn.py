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

    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        self.train_size = X.shape

    def predict(self, X: pd.DataFrame):
        predictions = []
        for i in range(X.shape[0]):
            distances = cdist(X.iloc[i : i + 1], self.X, metric="euclidean")
            nearest_indices = np.argsort(distances)[0][: self.k]
            nearest_labels = self.y.iloc[nearest_indices]
            most_common = Counter(nearest_labels).most_common(2)
            if len(most_common) == 0:
                predictions.append(1)
            elif len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                predictions.append(most_common[0][0])
            else:
                predictions.append(1)
        return np.array(predictions)

    def predict_proba(self, X: pd.DataFrame):
        probabilities = []
        for i in range(X.shape[0]):
            distances = cdist(X.iloc[i : i + 1], self.X, metric="euclidean")
            nearest_indices = np.argsort(distances)[0][: self.k]
            nearest_labels = self.y.iloc[nearest_indices]
            proba_class_1 = np.mean(nearest_labels == 1)
            probabilities.append(proba_class_1)
        return np.array(probabilities)
