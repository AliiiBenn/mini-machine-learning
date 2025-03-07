from collections.abc import Sequence
import math
from typing import TypeVar, overload

class MaxDictionnaryValue:
    def __init__(self, dictionnary: dict[float, float]) -> None:
        self.dictionnary = dictionnary

    @property
    def value(self) -> float:
        return max(self.dictionnary, key=lambda k: self.dictionnary[k])

class Mean:
    def __init__(self, data: Sequence[float]) -> None:
        self._data = data

    @property
    def value(self) -> float:
        return sum(self._data) / len(self._data)


class Variance:
    def __init__(self, data: Sequence[float]) -> None:
        self._data = data

    @property
    def value(self) -> float:
        mean = Mean(self._data).value
        return sum([(x - mean)**2 for x in self._data]) / len(self._data)


T = TypeVar('T')

class Labels(Sequence[T]):
    def __init__(self, *labels: T) -> None:
        self._labels = list(labels)

    def __len__(self) -> int:
        return len(self._labels)
    
    @overload
    def __getitem__(self, index: int) -> T:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[T]:
        ...

    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        return self._labels[index]
    

    @property
    def classes(self) -> Sequence[T]:
        return list(set(self._labels))

    @property
    def classes_prior(self) -> dict[T, float]:
        return {c: self._labels.count(c) / len(self._labels) for c in self.classes}

    
class Features(Sequence[Sequence[T]]):
    def __init__(self, *features: Sequence[T]) -> None:
        self._features = list(features)

    def __len__(self) -> int:
        return len(self._features)
    
    @overload
    def __getitem__(self, index: int) -> Sequence[T]:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Sequence[T]]:
        ...

    def __getitem__(self, index: int | slice) -> Sequence[T] | Sequence[Sequence[T]]:
        return self._features[index]
    


class FeatureCharacteristics:
    def __init__(self, cls: float, features: Features, indexes: Sequence[int], smoothing: float = 1e-9) -> None:
        self._cls = cls
        self._features = features
        self._indexes = indexes
        self._smoothing = smoothing

    @property
    def value(self) -> dict[float, dict[int, dict[str, float]]]:
        result: dict[float, dict[int, dict[str, float]]] = {}

        for i in range(len(self._features[0])):

            # Extract feature values for this class
            feature_values = [self._features[j][i] for j in self._indexes]
                
            # Calculate mean and variance with validation
            mean_value = Mean(feature_values).value
            variance_value = max(Variance(feature_values).value, self._smoothing)  # Ensure variance is not too small
                
            result[self._cls][i] = {
                'mean': mean_value,
                'variance': variance_value
            }

        return result

    

class FeatureLikelihood:
    def __init__(self, features: Features, labels: Labels, smoothing: float = 1e-9) -> None:
        self._features = features
        self._labels = labels
        self._smoothing = smoothing
        
    @property
    def value(self) -> dict[float, dict[int, dict[str, float]]]:
        result: dict[float, dict[int, dict[str, float]]] = {}
        
        for cls in self._labels.classes:
            result[cls] = {}
            # Get indices of samples belonging to class c
            indices = [j for j in range(len(self._labels)) if self._labels[j] == cls]
            
            if not indices:  # Handle empty class
                continue
                
            for i in range(len(self._features[0])):
                # Extract feature values for this class
                feature_values = [self._features[j][i] for j in indices]
                
                # Calculate mean and variance with validation
                mean_value = Mean(feature_values).value
                variance_value = max(Variance(feature_values).value, self._smoothing)  # Ensure variance is not too small
                
                result[cls][i] = {
                    'mean': mean_value,
                    'variance': variance_value
                }
                
        return result
        


class LogPrior:
    def __init__(self, class_prior: dict[float, float], c: float) -> None:
        self._class_prior = class_prior
        self._c = c

    @property
    def value(self) -> float:
        return math.log(self._class_prior[self._c])
    

class LogLikelihood:
    def __init__(self, x: list[float], c: float, feature_likelihood: dict[float, dict[int, dict[str, float]]], smoothing: float = 1e-9) -> None:
        self._x = x
        self._c = c
        self._feature_likelihood = feature_likelihood
        self._smoothing = smoothing 

    @property
    def value(self) -> float:
        log_likelihood = 0.0
        
        for feature_idx in range(len(self._x)):
            # Skip if feature not available for this class
            if feature_idx not in self._feature_likelihood[self._c]:
                continue
                
            mean = self._feature_likelihood[self._c][feature_idx]['mean']
            variance = self._feature_likelihood[self._c][feature_idx]['variance'] + self._smoothing
            
            # Calculate log likelihood directly to avoid underflow
            x_val = self._x[feature_idx]
            log_coef = -0.5 * math.log(2 * math.pi * variance)
            log_exp = -0.5 * ((x_val - mean) ** 2) / variance
            log_likelihood += log_coef + log_exp
        
        return log_likelihood



class PosteriorProbability:
    def __init__(self, x: list[float], c: float, class_prior: dict[float, float], 
                 feature_likelihood: dict[float, dict[int, dict[str, float]]], smoothing: float = 1e-9) -> None:
        self._x = x
        self._c = c
        self._class_prior = class_prior
        self._feature_likelihood = feature_likelihood
        self._smoothing = smoothing

    @property
    def value(self) -> float:
        # Use log probabilities for numerical stability
        log_prior = LogPrior(self._class_prior, self._c).value
        log_likelihood = LogLikelihood(self._x, self._c, self._feature_likelihood, self._smoothing).value
        
        return log_prior + log_likelihood
    

class LogPosteriorProbabilities:
    def __init__(self, x: list[float], classes: list[float], class_prior: dict[float, float], 
                 feature_likelihood: dict[float, dict[int, dict[str, float]]], smoothing: float = 1e-9) -> None:
        self._x = x
        self._classes = classes
        self._class_prior = class_prior
        self._feature_likelihood = feature_likelihood
        self._smoothing = smoothing

    @property
    def value(self) -> dict[float, float]:
        log_posteriors: dict[float, float] = {}

        for c in self._classes:
            if c not in self._feature_likelihood:
                continue  # Skip classes with no likelihood data
                
            log_posteriors[c] = PosteriorProbability(
                self._x, c, self._class_prior, self._feature_likelihood, self._smoothing
            ).value

        return log_posteriors
    
    @property
    def max_value(self) -> float:
        return max(self.value, key=lambda k: self.value[k])


class UnnormalizedPosteriors:
    def __init__(self, log_posteriors: LogPosteriorProbabilities) -> None:
        self._log_posteriors = log_posteriors

    @property
    def value(self) -> dict[float, float]:
        return {c: math.exp(log_post - self._log_posteriors.max_value) for c, log_post in self._log_posteriors.value.items()}


class PosteriorNormalizationFactor:
    def __init__(self, unnormalized_posteriors: UnnormalizedPosteriors) -> None:
        self._unnormalized_posteriors = unnormalized_posteriors

    @property
    def value(self) -> float:
        return sum(self._unnormalized_posteriors.value.values())
    

class NormalizedPosteriors:
    def __init__(self, unnormalized_posteriors: UnnormalizedPosteriors) -> None:
        self._unnormalized_posteriors = unnormalized_posteriors

    @property
    def value(self) -> dict[float, float]:
        return {
            c: value / PosteriorNormalizationFactor(self._unnormalized_posteriors).value 
            if PosteriorNormalizationFactor(self._unnormalized_posteriors).value > 0 
            else value 
            for c, value in self._unnormalized_posteriors.value.items()
        }


class PosteriorProbabilities:
    def __init__(self, x: list[float], classes: list[float], class_prior: dict[float, float], 
                 feature_likelihood: dict[float, dict[int, dict[str, float]]], smoothing: float = 1e-9) -> None:
        self._x = x
        self._classes = classes
        self._class_prior = class_prior
        self._feature_likelihood = feature_likelihood
        self._smoothing = smoothing

    @property
    def value(self) -> dict[float, float]:
        posteriors: dict[float, float] = {}
        
            
        unnormalized = UnnormalizedPosteriors(
            LogPosteriorProbabilities(
                self._x, 
                self._classes, 
                self._class_prior, 
                self._feature_likelihood, 
                self._smoothing
            )
        )
        
        normalization = PosteriorNormalizationFactor(unnormalized).value
        if normalization > 0:
            posteriors = NormalizedPosteriors(unnormalized).value
        else:
            posteriors = unnormalized.value
        
        return posteriors
    


    
class HighestPosteriorProbability:
    def __init__(self, posteriors: PosteriorProbabilities) -> None:
        self.posteriors = posteriors

    @property
    def value(self) -> float:
        return MaxDictionnaryValue(
            self.posteriors.value
        ).value




class NaiveBayes:
    def __init__(self, smoothing=1e-9):
        """
        Initialize the Naive Bayes classifier.

        Parameters:
        - smoothing (float): Additive smoothing parameter to handle zero probabilities.
        """
        self.smoothing = smoothing
        self.class_prior = {}
        self.feature_likelihood = {}
        self.classes = []
        self.log_priors = {}  # Store log priors for numerical stability

    def fit(self, X: list[list[float]], labels: Labels):
        """
        Fit the Naive Bayes classifier to the training data.

        Parameters:
        - X (list of lists): Training data features.
        - y (list): Training data labels.
        """
        # Input validation
        if not X or not labels or len(X) != len(labels):
            raise ValueError("Invalid input data")
            
        self.feature_likelihood = FeatureLikelihood(X, labels, self.smoothing).value
        
        # Compute log priors for numerical stability
        self.log_priors = {c: math.log(self.class_prior[c]) for c in self.classes}

    def predict(self, X):
        """
        Predict class labels for the given data.

        Parameters:
        - X (list of lists): Data features to predict.

        Returns:
        - list: Predicted class labels.
        """
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x: list[float]) -> float:
        posteriors = PosteriorProbabilities(
            x, self.classes, self.class_prior, self.feature_likelihood, self.smoothing
        )
        
        return HighestPosteriorProbability(posteriors).value




# Example usage:
if __name__ == '__main__':
    # Sample data
    X: list[list[float]] = [[2, 9], [2, 7], [3, 6], [4, 7], [2, 5], [6, 4], [7, 5], [8, 7], [9, 5], [7, 6]]
    labels = Labels(0, 0, 0, 0, 0, 1, 1, 1, 1, 1)

    # Create and train the Naive Bayes classifier
    nb = NaiveBayes(smoothing=1e-9)
    nb.fit(X, labels)

    # Make predictions
    predictions = nb.predict(X)
    print("Predictions:", predictions)
