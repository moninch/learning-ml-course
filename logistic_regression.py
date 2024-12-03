import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

eps = 1e-15


class MyLogReg:

    def __init__(
        self,
        n_iter=10,
        learning_rate=0.1,
        weights=None,
        metric=None,
        verbose=None,
        reg=None,
        l1_coef=0,
        l2_coef=0,
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None
        self.verbose = verbose
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __str__(self):
        return (
            f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):
        if "ones" not in X.columns:
            X.insert(loc=0, column="ones", value=1)
        num_features = X.shape[1]

        if self.weights is None:
            self.weights = np.ones(num_features)

        X = X.values
        y = y.values

        y_pred = self._sigmoid(X @ self.weights)
        log_loss = self._calculate_logloss(y, y_pred)
        metric_val = self._calculate_metric(y, y_pred)

        if verbose:
            print(f"start | loss: {log_loss:.2f}")

        for i in range(self.n_iter):
            if callable(self.learning_rate):
                curr_learning_rate = self.learning_rate(i + 1)
            else:
                curr_learning_rate = self.learning_rate

            y_pred = self._sigmoid(X @ self.weights)
            log_loss = self._calculate_logloss(y, y_pred)

            gradient = X.T @ (y_pred - y) / len(y)
            gradient += self._calculate_regularization_gradient()
            self.weights -= curr_learning_rate * gradient
            y_pred = self._sigmoid(X @ self.weights)
            metric_val = self._calculate_metric(y, y_pred)
            if verbose and (i + 1) % verbose == 0:
                print(
                    f"{i + 1} | loss: {log_loss:.2f} | {self.metric}: {metric_val:.2f}"
                )
            self.best_score = metric_val

    def _calculate_logloss(self, y_true, y_pred, eps=1e-15):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        log_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        log_loss += self._calculate_regularization()
        return log_loss

    def _calculate_metric(self, y_true, y_pred):
        y_pred_binary = (y_pred > 0.5).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))

        if self.metric == "accuracy":
            return (tp + tn) / (tp + tn + fp + fn + eps)

        elif self.metric == "precision":
            return tp / (tp + fp + eps)

        elif self.metric == "recall":
            return tp / (tp + fn + eps)

        elif self.metric == "f1":
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            return 2 * (precision * recall) / (precision + recall + eps)

        elif self.metric == "roc_auc":
            return roc_auc_score(y_true, np.round(y_pred, 10))

    def get_coef(self):
        return self.weights[1:]

    def _sigmoid(self, z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def predict_proba(self, X: pd.DataFrame):
        if "ones" not in X.columns:
            X.insert(loc=0, column="ones", value=1)
        return self._sigmoid(X @ self.weights)

    def predict(self, X: pd.DataFrame):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def get_best_score(self):
        return self.best_score

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


if __name__ == "__main__":
    X = pd.DataFrame(np.random.rand(400, 5))
    y = pd.Series(np.random.randint(0, 2, 400))

    model = MyLogReg(n_iter=50, learning_rate=0.1, metric="roc_auc", verbose=10)
    model.fit(X, y, verbose=10)
    print("Learned weights:", model.get_coef())
    proba = model.predict_proba(X)
    predictions = model.predict(X)
    print("Sum of predictions:", np.sum(predictions))
    print("Mean of probabilities:", np.mean(proba))
    print("Best score:", model.get_best_score())
