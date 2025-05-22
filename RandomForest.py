import numpy as np
from collections import Counter
from DecisionTree import DecisionTree
from typing import Optional, Literal

CriterionType = Optional[Literal["gini","entropy"]]

class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None, criterion: CriterionType = "gini"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.criterion = criterion
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            # Bootstrap aggregating (Bagging)
            ides = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[ides], y[ides]
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features, criterion=self.criterion)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        # print(f"Before: {tree_predictions}")
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        # print(f"After swap: {tree_predictions}")
        y_prediction = [Counter(row).most_common(1)[0][0] for row in tree_predictions]
        return np.array(y_prediction)
