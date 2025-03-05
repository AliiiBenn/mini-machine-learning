import math


class Mean:
    def __init__(self, data: list[float]) -> None:
        self._data = data

    @property
    def value(self) -> float:
        return sum(self._data) / len(self._data)


class Variance:
    def __init__(self, data: list[float]) -> None:
        self._data = data

    @property
    def value(self) -> float:
        mean = Mean(self._data).value
        return sum([(x - mean)**2 for x in self._data]) / len(self._data)


class GaussianLikelihood:
    def __init__(self, x: float, mean: Mean, variance: Variance) -> None:
        self._x = x 
        self._mean = mean
        self._variance = variance

    @property
    def value(self) -> float:
        mean = self._mean.value
        variance = self._variance.value

        if variance == 0:
            return 1e-9  # To avoid division by zero
        coeff = 1 / (math.sqrt(2 * math.pi * variance))
        exponent = math.exp(-((self._x - mean)**2 / (2 * variance)))
        return coeff * exponent
    


class NaiveBayes:
    def __init__(self, smoothing=1.0):
        """
        Initialize the Naive Bayes classifier.

        Parameters:
        - smoothing (float): Additive smoothing parameter to handle zero probabilities.
        """
        self.smoothing = smoothing
        self.class_prior = {}
        self.feature_likelihood = {}

    def _classes(self, y: list[float]) -> list[float]:
        return list(set(y))
    
    def _class_prior(self, y: list[float]) -> dict[float, float]:
        return {c: y.count(c) / len(y) for c in self._classes(y)}
    
    def _feature_likelihood(self, X: list[list[float]], y: list[float]) -> dict[float, dict[int, dict[str, float]]]:
        return {c: {i: {
            'mean': Mean(list(X[j][i] for j in range(len(y)) if y[j] == c)).value,
            'variance': Variance(list(X[j][i] for j in range(len(y)) if y[j] == c)).value
        } for i in range(len(X[0]))} for c in self._classes(y)}

    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier to the training data.

        Parameters:
        - X (list of lists): Training data features.
        - y (list): Training data labels.
        """
        self.classes = self._classes(y)
        self.class_prior = self._class_prior(y)
        self.feature_likelihood = self._feature_likelihood(X, y)

    def predict(self, X):
        """
        Predict class labels for the given data.
        
        Predict class labels for the given data.

        Parameters:
        - X (list of lists): Data features to predict.

        Returns:
        - list: Predicted class labels.
        """
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        """
        Predict the class label for a single sample.

        Parameters:
        - x (list): Single sample features.

        Returns:
        - object: Predicted class label.
        """
        posteriors = {}
        for c in self.classes:
            # Calculate posterior probability for each class
            prior = self.class_prior[c]
            likelihood = 1
            for feature_idx in range(len(x)):
                mean = self.feature_likelihood[c][feature_idx]['mean']
                variance = self.feature_likelihood[c][feature_idx]['variance']
                # Gaussian likelihood
                likelihood *= GaussianLikelihhood(x[feature_idx], mean, variance).value
            posterior = prior * likelihood
            posteriors[c] = posterior

        if not posteriors:
            return None  # Or a default class if appropriate

        # Return the class with the highest posterior probability
        return max(posteriors, key=posteriors.get, default=None)



# Example usage:
if __name__ == '__main__':
    # Sample data
    X = [[2, 9], [2, 7], [3, 6], [4, 7], [2, 5], [6, 4], [7, 5], [8, 7], [9, 5], [7, 6]]
    y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    # Create and train the Naive Bayes classifier
    nb = NaiveBayes(smoothing=1.0)
    nb.fit(X, y)

    # Make predictions
    predictions = nb.predict(X)
    print("Predictions:", predictions)
