class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature      # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left: Node = left            # Left subtree (<= threshold)
        self.right: Node = right          # Right subtree (> threshold)
        self.value = value          # Class label if it's a leaf node

    def is_leaf_node(self) -> bool:
        return self.value is not None