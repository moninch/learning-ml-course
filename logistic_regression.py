import pandas as pd
import numpy as np

eps = 1e-15


class MyLogReg:

    def __init__(self, n_iter=10, learning_rate=0.1, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        return (
            f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):
        X.insert(loc=0, column="ones", value=1)
        num_features = X.shape[1]

        if self.weights is None:
            self.weights = np.ones(num_features)

        X = X.values
        y = y.values

        y_pred = self._sigmoid(X @ self.weights)
        log_loss = self._calculate_logloss(y, y_pred)
        if verbose:
            print(f"start | loss: {log_loss:.2f}")

        for i in range(self.n_iter):
            y_pred = self._sigmoid(X @ self.weights)
            log_loss = self._calculate_logloss(y, y_pred)

            gradient = X.T @ (y_pred - y) / len(y)
            self.weights -= self.learning_rate * gradient

            if verbose and (i + 1) % verbose == 0:
                print(f"{i + 1} | loss: {log_loss:.2f}")

    def _calculate_logloss(self, y_true, y_pred, eps=1e-15):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def get_coef(self):
        return self.weights[1:]

    def _sigmoid(self, z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
