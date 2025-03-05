from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Generic, TypeVar, List, Any

T = TypeVar('T')


class MatrixSum:
    def __init__(self, matrix: Sequence[Sequence[int]]):
        self._matrix = matrix

    @property
    def value(self) -> int:
        return sum(sum(row) for row in self._matrix)
    

class MatrixSumOfDiagonal:
    def __init__(self, matrix: Sequence[Sequence[int]]):
        self._matrix = matrix

    @property
    def value(self) -> int:
        return sum(self._matrix[i][i] for i in range(len(self._matrix)))


class ConfusionMatrixAccuracy:
    """
    Calculates the accuracy of the confusion matrix.

    Accuracy is the proportion of true results (both true positives and true negatives)
    among the total number of cases examined.
    """
    def __init__(self, confusion_matrix: ConfusionMatrix[T]):
        """
        Initializes the ConfusionMatrixAccuracy with a confusion matrix.

        Args:
            confusion_matrix (ConfusionMatrix[T]): The confusion matrix to calculate accuracy from.
        """
        self._confusion_matrix = confusion_matrix


    @property
    def value(self) -> float:
        """
        Calculates the accuracy value.

        Returns:
            float: The accuracy of the confusion matrix.
        """
        MATRIX = self._confusion_matrix.value

        DIAGONAL_SUM = MatrixSumOfDiagonal(MATRIX).value 
        MATRIX_SUM = MatrixSum(MATRIX).value

        IS_MATRIX_NULL = MATRIX_SUM == 0
        ACCURACY = DIAGONAL_SUM / MATRIX_SUM if not IS_MATRIX_NULL else 0.0

        return ACCURACY


class ConfusionMatrixLabelPrecision:
    """
    Calculates the precision for a specific label in the confusion matrix.

    Precision is the proportion of predicted positives that are actually positive.
    It indicates how well the model avoids false positives for a particular class.
    """
    def __init__(self, confusion_matrix: ConfusionMatrix[T], *, label: T) -> None:
        """
        Initializes the ConfusionMatrixLabelPrecision with a confusion matrix and a label.

        Args:
            confusion_matrix (ConfusionMatrix[T]): The confusion matrix to calculate precision from.
            label (T): The label for which to calculate precision.
        """
        self._confusion_matrix = confusion_matrix
        self._label = label

    @property
    def value(self) -> float:
        """
        Calculates the precision value for the specified label.

        Returns:
            float: The precision for the label.
        """
        TRUE_POSITIVES: int = self._confusion_matrix.true_positive(self._label)
        FALSE_POSITIVES: int = self._confusion_matrix.false_positive(self._label)
        
        PRECISION_DIVISOR: int = TRUE_POSITIVES + FALSE_POSITIVES
        
        return TRUE_POSITIVES / PRECISION_DIVISOR if PRECISION_DIVISOR > 0 else 0.0



class ConfusionMatrixLabelRecall:
    """
    Calculates the recall for a specific label in the confusion matrix.

    Recall is the proportion of actual positives that are correctly identified.
    It indicates how well the model avoids false negatives for a particular class.
    """
    def __init__(self, confusion_matrix: ConfusionMatrix[T], *, label: T) -> None:
        """
        Initializes the ConfusionMatrixLabelRecall with a confusion matrix and a label.

        Args:
            confusion_matrix (ConfusionMatrix[T]): The confusion matrix to calculate recall from.
            label (T): The label for which to calculate recall.
        """
        self._confusion_matrix = confusion_matrix
        self._label = label

    @property
    def value(self) -> float:
        """
        Calculates the recall value for the specified label.

        Returns:
            float: The recall for the label.
        """
        TRUE_POSITIVES: int = self._confusion_matrix.true_positive(self._label)
        FALSE_NEGATIVES: int = self._confusion_matrix.false_negative(self._label)
        
        RECALL_DIVISOR: int = TRUE_POSITIVES + FALSE_NEGATIVES
        
        return TRUE_POSITIVES / RECALL_DIVISOR if RECALL_DIVISOR > 0 else 0.0


class ConfusionMatrixLabelSpecificity:
    """
    Calculates the specificity for a specific label in the confusion matrix.

    Specificity is the proportion of actual negatives that are correctly identified.
    It measures the ability of the model to avoid false positives for the negative class.
    """
    def __init__(self, confusion_matrix: ConfusionMatrix[T], *, label: T) -> None:
        """
        Initializes the ConfusionMatrixLabelSpecificity with a confusion matrix and a label.

        Args:
            confusion_matrix (ConfusionMatrix[T]): The confusion matrix to calculate specificity from.
            label (T): The label for which to calculate specificity.
        """
        self._confusion_matrix = confusion_matrix
        self._label = label

    @property
    def value(self) -> float:
        """
        Calculates the specificity value for the specified label.

        Returns:
            float: The specificity for the label.
        """
        TRUE_NEGATIVES: int = self._confusion_matrix.true_negative(self._label)
        FALSE_POSITIVES: int = self._confusion_matrix.false_positive(self._label)
        
        SPECIFICITY_DIVISOR: int = TRUE_NEGATIVES + FALSE_POSITIVES
        
        return TRUE_NEGATIVES / SPECIFICITY_DIVISOR if SPECIFICITY_DIVISOR > 0 else 0.0
    

class ConfusionMatrixLabelF1Score:
    """
    Calculates the F1-score for a specific label in the confusion matrix.

    The F1-score is the harmonic mean of precision and recall, providing a balanced
    measure of a test's accuracy. It considers both false positives and false negatives.
    """
    def __init__(self, confusion_matrix: ConfusionMatrix[T], *, label: T) -> None:
        """
        Initializes the ConfusionMatrixLabelF1Score with a confusion matrix and a label.

        Args:
            confusion_matrix (ConfusionMatrix[T]): The confusion matrix to calculate F1-score from.
            label (T): The label for which to calculate F1-score.
        """
        self._confusion_matrix = confusion_matrix
        self._label = label

    @property
    def value(self) -> float:
        """
        Calculates the F1-score value for the specified label.

        Returns:
            float: The F1-score for the label.
        """
        PRECISION: float = self._confusion_matrix.precision(self._label)
        RECALL: float = self._confusion_matrix.recall(self._label)
        
        F1_SCORE_DIVISOR: float = PRECISION + RECALL
        
        return 2 * (PRECISION * RECALL) / F1_SCORE_DIVISOR if F1_SCORE_DIVISOR > 0 else 0.0


class ConfusionMatrix(Generic[T]):
    def __init__(self, actual: Sequence[T], predicted: Sequence[T], labels: Sequence[T]):
        """
        Initializes the ConfusionMatrix.

        Args:
            actual (list): List of actual class labels.
            predicted (list): List of predicted class labels.
            labels (list): List of possible class labels.
        """
        self.actual = actual
        self.predicted = predicted
        self.labels = labels

    @property
    def value(self) -> List[List[int]]:
        """
        Creates the confusion matrix.

        Returns:
            List[List[int]]: The confusion matrix.
        """
        matrix: List[List[int]] = [[0 for _ in self.labels] for _ in self.labels]
        for a, p in zip(self.actual, self.predicted):
            matrix[self.labels.index(a)][self.labels.index(p)] += 1
        return matrix
    

    @property
    def accuracy(self) -> float:
        return ConfusionMatrixAccuracy(self).value
    

    def true_positive(self, label: T) -> int:
        class_index: int = self.labels.index(label)
        return self.value[class_index][class_index]

    def true_negative(self, label: T) -> int:
        class_index: int = self.labels.index(label)
        tn: int = sum(
            sum(
                self.value[i][j]
                for j in range(len(self.labels))
                if j != class_index
            )
            for i in range(len(self.labels))
            if i != class_index
        )
        return tn

    def false_positive(self, label: T) -> int:
        class_index: int = self.labels.index(label)
        fp: int = sum(
            self.value[i][class_index]
            for i in range(len(self.labels))
            if i != class_index
        )
        return fp

    def false_negative(self, label: T) -> int:
        class_index: int = self.labels.index(label)
        fn: int = sum(self.value[class_index][i] for i in range(len(self.labels))) - self.true_positive(label)
        return fn

    def precision(self, label: T) -> float:
        return ConfusionMatrixLabelPrecision(self, label=label).value


    def recall(self, label: T) -> float:
        return ConfusionMatrixLabelRecall(self, label=label).value

    def specificity(self, label: T) -> float:
        return ConfusionMatrixLabelSpecificity(self, label=label).value

    def f1_score(self, label: T) -> float:
        return ConfusionMatrixLabelF1Score(self, label=label).value

    def __str__(self) -> str:
        return "\n".join(str(row) for row in self.value)







if __name__ == '__main__':
    # Example Usage
    actual: List[str] = ['dog', 'cat', 'dog', 'dog', 'cat', 'cat', 'dog', 'cat']
    predicted: List[str] = ['dog', 'cat', 'cat', 'dog', 'cat', 'dog', 'dog', 'cat']
    labels: List[str] = ['dog', 'cat']

    cm: ConfusionMatrix[str] = ConfusionMatrix(actual, predicted, labels)
    print(cm)
    print(f"Accuracy: {cm.accuracy}")
    print(f"Precision (dog): {cm.precision('dog')}")
    print(f"Recall (dog): {cm.recall('dog')}")
    print(f"Specificity (dog): {cm.specificity('dog')}")
    print(f"F1 Score (dog): {cm.f1_score('dog')}")
    print(f"Precision (cat): {cm.precision('cat')}")
    print(f"Recall (cat): {cm.recall('cat')}")
    print(f"Specificity (cat): {cm.specificity('cat')}")
    print(f"F1 Score (cat): {cm.f1_score('cat')}")
