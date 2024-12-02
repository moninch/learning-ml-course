import pandas as pd
import numpy as np


class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None

    def __str__(self):
        return (
            f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        X.insert(loc=0, column="ones", value=1)
        num_features = X.shape[1]
        if self.weights is None:
            self.weights = np.ones(num_features)
        X = X.values
        y = y.values

        for _ in range(self.n_iter):
            y_pred = X @ self.weights
            mse = np.mean((y - y_pred) ** 2)
            gradient = (2 / len(y)) * X.T @ (y_pred - y)
            self.weights -= self.learning_rate * gradient
            if verbose and (_ + 1) % 100 == 0:
                print(f"Iteration {_ + 1}/{self.n_iter}, MSE: {mse}")

    def get_coef(self):
        return self.weights[1:]


a = MyLineReg(n_iter=100, learning_rate=0.1)
print(a)
