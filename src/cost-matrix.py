from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, TypeVar, List, Any

T = TypeVar('T')


class CostMatrix(Generic[T]):
    """
    Represents a cost matrix for evaluating classification performance,
    where each element c_ij represents the cost associated with classifying
    an example from the actual class i as the predicted class j.
    """

    def __init__(self, costs: List[List[float]], labels: Sequence[T]):
        """
        Initializes the CostMatrix with a cost matrix and a list of labels.

        Args:
            costs (List[List[float]]): A 2D list representing the cost matrix,
                                         where costs[i][j] is the cost of classifying
                                         an instance of class i as class j.
            labels (Sequence[T]): A list of possible class labels.
        """
        self.costs = costs
        self.labels = labels

        # Validate the cost matrix dimensions
        if len(costs) != len(labels) or any(len(row) != len(labels) for row in costs):
            raise ValueError("Cost matrix dimensions must match the number of labels.")

    def get_cost(self, actual: T, predicted: T) -> float:
        """
        Returns the cost associated with classifying an instance of the actual class
        as the predicted class.

        Args:
            actual (T): The actual class label.
            predicted (T): The predicted class label.

        Returns:
            float: The cost of the misclassification.
        """
        actual_index = self.labels.index(actual)
        predicted_index = self.labels.index(predicted)
        return self.costs[actual_index][predicted_index]

    def calculate_total_cost(self, actual: Sequence[T], predicted: Sequence[T]) -> float:
        """
        Calculates the total cost of a series of classifications, given the actual
        and predicted labels.

        Args:
            actual (Sequence[T]): A sequence of actual class labels.
            predicted (Sequence[T]): A sequence of predicted class labels.

        Returns:
            float: The total cost of the classifications.
        """
        if len(actual) != len(predicted):
            raise ValueError("Actual and predicted label sequences must have the same length.")

        total_cost = 0.0
        for a, p in zip(actual, predicted):
            total_cost += self.get_cost(a, p)
        return total_cost

    def __str__(self) -> str:
        """
        Returns a string representation of the cost matrix.
        """
        # Determine the maximum width needed for the labels and costs
        max_label_width = max(len(str(label)) for label in self.labels)
        max_cost_width = max(max(len(f"{cost:.2f}") for cost in row) for row in self.costs)

        # Construct the header
        header = " " * (max_label_width + 1) + "| " + " | ".join(f"{str(label):^{max_cost_width}}" for label in self.labels) + " |"
        separator = "-" * (max_label_width + 1) + "+-" + "-+-".join("-" * max_cost_width for _ in self.labels) + "-+"
        rows = [header, separator]

        # Construct the rows
        for i, label in enumerate(self.labels):
            cost_strings = [f"{cost:.2f}".center(max_cost_width) for cost in self.costs[i]]
            row = f"{str(label):>{max_label_width}} | " + " | ".join(cost_strings) + " |"
            rows.append(row)

        return "\n".join(rows)


if __name__ == '__main__':
    # Example Usage
    labels: List[str] = ['dog', 'cat']
    costs: List[List[float]] = [
        [0.0, 1.0],  # Cost of classifying dog as dog is 0, as cat is 1
        [2.0, 0.0]   # Cost of classifying cat as dog is 2, as cat is 0
    ]

    cm: CostMatrix[str] = CostMatrix(costs, labels)
    print(cm)

    actual: List[str] = ['dog', 'cat', 'dog', 'cat', 'cat']
    predicted: List[str] = ['dog', 'cat', 'cat', 'dog', 'dog']

    total_cost: float = cm.calculate_total_cost(actual, predicted)
    print(f"Total cost: {total_cost}")  # Expected total cost: 2.0
