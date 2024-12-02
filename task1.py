import numpy as np
import pandas as pd


class MyLineReg:
    def __init__(
        self,
        n_iter=100,
        learning_rate=0.1,
        metric=None,
        reg=None,
        l1_coef=0,
        l2_coef=0,
        sgd_sample=None,
        random_state=42,
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.best_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return (
            f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=True):
        X.insert(loc=0, column="ones", value=1)
        num_features = X.shape[1]

        if self.weights is None:
            self.weights = np.ones(num_features)

        X = X.values
        y = y.values

        for i in range(self.n_iter):
            y_pred = X @ self.weights
            mse = np.mean((y - y_pred) ** 2)

            loss = mse + self._calculate_regularization()

            gradient = (2 / len(y)) * X.T @ (y_pred - y)

            gradient += self._calculate_regularization_gradient()
            if callable(self.learning_rate):
                curr_learning_rate = self.learning_rate(i + 1)
            else:
                curr_learning_rate = self.learning_rate
            self.weights -= curr_learning_rate * gradient

            y_pred = X @ self.weights
            metric_value = self._calculate_metric(y, y_pred)
            if verbose and (i + 1) % verbose == 0:
                print(f"{i + 1} | loss: {loss:.2f} | {self.metric}: {metric_value:.2f}")
            self.best_score = metric_value

    def _calculate_metric(self, y_true, y_pred):
        if self.metric == "mae":
            return np.mean(abs(y_true - y_pred))
        elif self.metric == "mse":
            return np.mean((y_true - y_pred) ** 2)
        elif self.metric == "rmse":
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif self.metric == "mape":
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        elif self.metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)
        else:
            return None

    def _calculate_regularization(self):
        if self.reg == "l1":
            return self.l1_coef * np.sum(np.abs(self.weights))

        elif self.reg == "l2":
            return self.l2_coef * np.sum(self.weights**2)

        elif self.reg == "elasticnet":
            return self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum(
                self.weights**2
            )
        else:
            return 0

    def _calculate_regularization_gradient(self):
        if self.reg == "l1":
            return self.l1_coef * np.sign(self.weights)

        if self.reg == "l2":
            return self.l2_coef * 2 * self.weights

        elif self.reg == "elasticnet":
            return (
                self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
            )
        else:
            return 0

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X: pd.DataFrame):
        X.insert(loc=0, column="ones", value=1)
        return X @ self.weights

    def get_best_score(self):
        return self.best_score
