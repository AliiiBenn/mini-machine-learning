from typing import Optional, Sequence, Tuple, Union
import numpy as np
from .gini import GiniImpurity, GiniSplit

class Node:
    """
    A node in the decision tree.
    
    Attributes:
        feature_index: Index of the feature used for splitting
        threshold: Value used for splitting
        left: Left child node
        right: Right child node
        value: Predicted value for leaf nodes
        is_leaf: Whether this is a leaf node
    """
    def __init__(
        self,
        *,
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None,
        value: Optional[float] = None
    ) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = value is not None







class DecisionTreeClassifier:
    """
    A decision tree classifier using Gini impurity for splits.
    
    Parameters:
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum number of samples required to split
        min_samples_leaf: Minimum number of samples required in a leaf
    """
    def __init__(
        self,
        *,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root: Optional[Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """
        Build the decision tree classifier.
        
        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        """
        self.n_classes_ = len(np.unique(y))
        self.root = self._grow_tree(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class for X.
        
        Parameters:
            X: The input samples of shape (n_samples, n_features)
            
        Returns:
            y: The predicted classes of shape (n_samples,)
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int = 0
    ) -> Node:
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return Node(value=float(np.argmax(np.bincount(y.astype(int)))))

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            return Node(value=float(np.argmax(np.bincount(y.astype(int)))))

        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Check min_samples_leaf constraint
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return Node(value=float(np.argmax(np.bincount(y.astype(int)))))

        # Create child nodes
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left,
            right=right
        )

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float]]:
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                gini = GiniSplit(y[left_mask], y[right_mask]).value
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _traverse_tree(self, x: np.ndarray, node: Optional[Node]) -> float:
        if node is None or node.is_leaf:
            return float(node.value if node else 0.0)

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
