import pandas as pd
import numpy as np


class MyTreeReg:

    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.tree = None

    def __str__(self):
        return f"MyTreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"

    def _mse(self, y, y_left, y_right):
        mse_total = np.mean((y - np.mean(y)) ** 2)
        mse_left = np.mean((y_left - np.mean(y_left)) ** 2)
        mse_right = np.mean((y_right - np.mean(y_right)) ** 2)
        p_left = len(y_left) / len(y)
        p_right = len(y_right) / len(y)
        return mse_total - (p_left * mse_left + p_right * mse_right)

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        col_name = None
        split_value = None
        gain = 0

        for col in X.columns:

            unique_values = np.sort(X[col].unique())
            split_values = (unique_values[:-1] + unique_values[1:]) / 2
            for split_value_temp in split_values:
                left_mask = X[col] <= split_value_temp
                right_mask = X[col] > split_value_temp

                y_left = y[left_mask]
                y_right = y[right_mask]

                if (
                    len(y_left) < self.min_samples_split
                    or len(y_right) < self.min_samples_split
                ):
                    continue

                gain_temp = self._mse(y, y_left, y_right)
                if gain_temp > gain:
                    gain = gain_temp
                    col_name = col
                    split_value = split_value_temp

        return col_name, split_value, gain
