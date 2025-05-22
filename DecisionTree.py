import numpy as np
from typing import Optional, Literal
from collections import Counter
from Node import Node

CriterionType = Optional[Literal["gini","entropy"]]

class DecisionTree:
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2, n_features: Optional[int] = None, criterion: CriterionType = "gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.criterion = criterion
        self.tree: Optional[Node] = None

    def fit(self, X, y):
        self.n_features = self.n_features or X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_ides = np.random.choice(n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_ides)

        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_ides = X[:, best_feat] < best_thresh
        right_ides = ~left_ides

        left = self._grow_tree(X[left_ides], y[left_ides], depth + 1)
        right = self._grow_tree(X[right_ides], y[right_ides], depth + 1)
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, feat_ides):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_ides:
            thresholds = np.unique(X[:, feat_idx])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feat_idx], threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, feature_column, threshold):
        gain = -1

        left_ides = feature_column < threshold
        right_ides = ~left_ides

        y_left = y[left_ides]
        y_right = y[right_ides]

        n_parent = len(y)
        n_left, n_right = len(y_left), len(y_right)

        weight_left = n_left / n_parent
        weight_right = n_right / n_parent

        if n_left == 0 or n_right == 0:
            gain = 0

        if self.criterion == "entropy":
            parent_entropy = self._entropy(y)
            e_left, e_right = self._entropy(y_left), self._entropy(y_right)
            children_entropy = (weight_left * e_left) + (weight_right * e_right)
            gain = parent_entropy - children_entropy
        elif self.criterion == "gini":
            parent_gini = self._gini(y)
            g_left, g_right = self._gini(y_left), self._gini(y_right)
            children_gini = (weight_left * g_left) + (weight_right * g_right)
            gain = parent_gini - children_gini

        return gain

    def _gini(self, y) -> float:
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)