from collections import Counter
from collections.abc import Sequence
from typing import Final


class GiniImpurity:
    """
    Calculates the Gini impurity of a node in a decision tree.
    
    Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled
    if it was randomly labeled according to the distribution of labels in the subset. It reaches its minimum
    (zero) when all cases in the node fall into a single target category.
    
    The formula for Gini impurity is:
    1 - Σ(p_i)^2 where p_i is the proportion of samples that belong to class i.
    
    For binary classification:
    - A pure node (all samples of same class) has Gini = 0
    - An equal 50-50 split between classes has Gini = 0.5
    - Maximum Gini impurity is 0.5 for binary classification
    """
    _y: Final[Sequence[float]]

    def __init__(self, y: Sequence[float]) -> None:
        self._y = y

    @property 
    def value(self) -> float:
        if len(self._y) == 0:
            return 0.0
        
        COUNTS = Counter(self._y)
        N_SAMPLES = len(self._y)
        PROBABILITIES = [COUNT/N_SAMPLES for COUNT in COUNTS.values()]

        return 1 - sum(map(lambda x: x*x, PROBABILITIES))


class GiniSplit:
    """
    Calculates the weighted Gini impurity for a binary split in a decision tree.
    
    When a decision tree makes a split, it divides the data into two groups (left and right).
    GiniSplit evaluates the quality of this split by computing a weighted average of the Gini
    impurity of both resulting groups. The weights are proportional to the size of each group.
    
    The formula for Gini split is:
    (n_left/n_total * gini_left) + (n_right/n_total * gini_right)
    where:
    - n_left, n_right: number of samples in left and right groups
    - n_total: total number of samples
    - gini_left, gini_right: Gini impurity of left and right groups
    
    Properties of Gini split:
    - A perfect split (each group pure) has value = 0
    - Worse splits have higher values
    - Used in decision trees to evaluate potential splitting points
    - The best split minimizes this weighted Gini impurity
    
    Example:
    - If split creates pure groups (all same class), GiniSplit = 0
    - If split creates equally impure groups, GiniSplit will be high
    - Unbalanced splits (very different group sizes) are weighted accordingly
    """
    _y_left: Final[Sequence[float]]
    _y_right: Final[Sequence[float]]

    def __init__(self, y_left: Sequence[float], y_right: Sequence[float]) -> None:
        self._y_left = y_left 
        self._y_right = y_right

    @property
    def value(self) -> float:
        n_left = len(self._y_left)
        n_right = len(self._y_right)
        n_total = n_left + n_right
        
        if n_total == 0:
            return 0.0
        
        # Calculate weighted average of Gini impurity
        gini_left = GiniImpurity(self._y_left).value
        gini_right = GiniImpurity(self._y_right).value
        
        weighted_gini = (n_left/n_total * gini_left + 
                        n_right/n_total * gini_right)
        
        return weighted_gini
