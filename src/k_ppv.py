from collections.abc import Sequence
from typing import Generic, TypeVar
import math
from collections import Counter

from features import Features
from labels import Labels


T = TypeVar('T', bound=float)
class EuclideanDistance:
    _x1: Sequence[float]
    _x2: Sequence[float]

    def __init__(self, x1: Sequence[float], x2: Sequence[float]) -> None:
        self._x1 = x1
        self._x2 = x2

    @property
    def value(self) -> float:
        distance: float = 0
        for i in range(len(self._x1)):
            distance += (self._x1[i] - self._x2[i]) ** 2
        return math.sqrt(distance)


class EuclideanDistances:
    def __init__(self, x1: Sequence[Sequence[float]], x2: Sequence[float]) -> None:
        self._x1 = x1
        self._x2 = x2

    @property
    def value(self) -> Sequence[float]:
        return [EuclideanDistance(x1, self._x2).value for x1 in self._x1]

class KNearestIndices:
    def __init__(self, distances: EuclideanDistances, *, k: int) -> None:
        self._distances = distances
        self._k = k

    @property
    def value(self) -> Sequence[int]:
        return sorted(range(len(self._distances.value)), key=lambda i: self._distances.value[i])[:self._k]


class KNearestLabels(Generic[T]):
    def __init__(self, labels: Labels[T], distances: EuclideanDistances, *, k: int) -> None:
        self._labels = labels
        self._k_nearest_indices = KNearestIndices(distances, k=k)

    @property
    def value(self) -> Sequence[T]:
        return list(map(lambda i: self._labels[i], self._k_nearest_indices.value))
    
    def most_common(self) -> T:
        return Counter(self.value).most_common(1)[0][0]
    


class KNN(Generic[T]):
    def __init__(self, neighbors: int = 3) -> None:
        self.neighbors: int = neighbors
        self.features: Features[T] | None = None
        self.labels: Labels[T] | None = None

    def fit(self, features: Features[T], labels: Labels[T]):
        """
        Fit the KNN classifier to the training data.

        Parameters:
        - features (list of lists): Training data features.
        - labels (list): Training data labels.
        """
        self.features = features
        self.labels = labels

    def predict(self, features: Features[T]) -> Labels[T]:
        """
        Predict class labels for the given data.

        Parameters:
        - features (list of lists): Data features to predict.

        Returns:
        - list: Predicted class labels.
        """
        predictions = [self._predict(feature) for feature in features]
        return Labels(*predictions)

    def _predict(self, feature: Sequence[float]) -> T:
        """
        Predict the class label for a single sample.

        Parameters:
        - feature (list): Single sample features.

        Returns:
        - T: Predicted class label.
        """
        if self.features is None:
            raise ValueError("The fit method must be called before predict.")

        if self.labels is None:
            raise ValueError("The fit method must be called before predict.")

        return KNearestLabels(
            labels=self.labels,
            distances=EuclideanDistances(x1=self.features, x2=feature),
            k=self.neighbors
        ).most_common()


# Example usage:
if __name__ == '__main__':
    # Sample data
    X_train = [[float(x) for x in row] for row in [[2, 9], [2, 7], [3, 6], [4, 7], [2, 5], [6, 4], [7, 5], [8, 7], [9, 5], [7, 6]]]
    y_train = [float(y) for y in [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
    X_test = [[float(x) for x in row] for row in [[3, 7], [7, 4]]]

    # Create and train the KNN classifier
    knn: KNN[float] = KNN(neighbors=3)
    knn.fit(Features(*X_train), Labels(*y_train))

    # Make predictions
    predictions = knn.predict(Features(*X_test))
    print("Predictions:", predictions)

