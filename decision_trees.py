import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


class MyTreeClf:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.tree = None
        self.leafs_cnt = 0

    def __str__(self):
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split }, max_leafs={self.max_leafs}"

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        ig = -np.inf
        col_name = None
        split_value = None

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

                ig_temp = self._information_gain(y, y_left, y_right)
                if ig_temp > ig:
                    ig = ig_temp
                    col_name = col
                    split_value = split_value_temp

        return col_name, split_value, ig

    def _information_gain(self, y, y_left, y_right):
        H_y = self._entropy(y)
        H_left = self._entropy(y_left)
        H_right = self._entropy(y_right)
        p_left = len(y_left) / len(y)
        p_right = len(y_right) / len(y)
        return H_y - (p_left * H_left + p_right * H_right)

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        else:
            counts = np.bincount(y)
            probabilities = counts / len(y)
            return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X.copy()
        self.y = y.copy()
        self.tree = self._build_tree(self.X, self.y, depth=0)

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int):
        # Проверка на условия остановки
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or self.leafs_cnt >= self.max_leafs
        ):
            self.leafs_cnt += 1
            return {"class": np.bincount(y).argmax()}

        # Проверка на однородность классов
        if len(y.unique()) == 1:
            self.leafs_cnt += 1
            return {"class": y.unique()[0]}

        # Поиск лучшего сплита
        col_name, split_value, ig = self.get_best_split(X, y)

        # Если не найден лучший сплит, возвращаем лист
        if col_name is None or ig == 0:
            self.leafs_cnt += 1
            return {"class": np.bincount(y).argmax()}

        # Разделение данных
        left_mask = X[col_name] <= split_value
        right_mask = X[col_name] > split_value

        y_left = y[left_mask]
        y_right = y[right_mask]

        # Если одна из подвыборок пуста, возвращаем лист
        if len(y_left) == 0 or len(y_right) == 0:
            self.leafs_cnt += 1
            return {"class": np.bincount(y).argmax()}

        # Рекурсивное построение левого и правого поддеревьев
        left_tree = self._build_tree(X[left_mask], y_left, depth + 1)
        right_tree = self._build_tree(X[right_mask], y_right, depth + 1)

        return {
            "col_name": col_name,
            "split_value": split_value,
            "left": left_tree,
            "right": right_tree,
        }

    def _print_tree(self, node, depth=0):
        if "class" in node:
            print("  " * depth + f'leaf = {node["class"]}')
        else:
            print("  " * depth + f'{node["col_name"]} > {node["split_value"]}')
            self._print_tree(node["left"], depth + 1)
            self._print_tree(node["right"], depth + 1)

    def print_tree(self):
        if self.tree is not None:
            self._print_tree(self.tree)
        else:
            print("Дерево еще не построено.")

    def _predict_proba_row(self, row, node):
        if "class" in node:
            return node["class"]
        if row[node["col_name"]] <= node["split_value"]:
            return self._predict_proba_row(row, node["left"])
        else:
            return self._predict_proba_row(row, node["right"])

    def predict_proba(self, X: pd.DataFrame):
        return [self._predict_proba_row(row, self.tree) for _, row in X.iterrows()]

    def predict(self, X: pd.DataFrame):
        probas = self.predict_proba(X)
        return [1 if p > 0.5 else 0 for p in probas]


iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Пример использования
tree = MyTreeClf(max_depth=3, min_samples_split=2, max_leafs=1)
tree.fit(X, y)
print(f"Количество листьев: {tree.leafs_cnt}")
tree.print_tree()
