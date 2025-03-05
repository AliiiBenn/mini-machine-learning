import math
from collections import Counter

class KNN:
    def __init__(self, k: int):
        """
        Initialize the KNN classifier.

        Parameters:
        - k (int): The number of neighbors to consider.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train: list[list[float]], y_train: list[float]):
        """
        Fit the KNN classifier to the training data.

        Parameters:
        - X_train (list of lists): Training data features.
        - y_train (list): Training data labels.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test: list[list[float]]) -> list[float]:
        """
        Predict class labels for the given data.

        Parameters:
        - X_test (list of lists): Data features to predict.

        Returns:
        - list: Predicted class labels.
        """
        predictions = [self._predict(x) for x in X_test]
        return predictions

    def _predict(self, x: list[float]) -> float:
        """
        Predict the class label for a single sample.

        Parameters:
        - x (list): Single sample features.

        Returns:
        - float: Predicted class label.
        """
        # Calculate distances from the input sample to all training samples
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get the indices of the k-nearest neighbors
        k_nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]

        # Get the labels of the k-nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        # Return the most common class label among the k-nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1: list[float], x2: list[float]) -> float:
        """
        Calculate the Euclidean distance between two samples.

        Parameters:
        - x1 (list): First sample features.
        - x2 (list): Second sample features.

        Returns:
        - float: Euclidean distance between the two samples.
        """
        distance = 0
        for i in range(len(x1)):
            distance += (x1[i] - x2[i]) ** 2
        return math.sqrt(distance)

# Example usage:
if __name__ == '__main__':
    # Sample data
    X_train = [[2, 9], [2, 7], [3, 6], [4, 7], [2, 5], [6, 4], [7, 5], [8, 7], [9, 5], [7, 6]]
    y_train = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    X_test = [[3, 7], [7, 4]]

    # Create and train the KNN classifier
    knn = KNN(k=3)
    knn.fit(X_train, y_train)

    # Make predictions
    predictions = knn.predict(X_test)
    print("Predictions:", predictions)
