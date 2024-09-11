import numpy as np
from typing import Self

"""
This is a suggested template and you do not need to follow it. You can change any part of it to fit your needs.
There are some helper functions that might be useful to implement first.
At the end there is some test code that you can use to test your implementation on synthetic data by running this file.
"""

def count(y: np.ndarray) -> np.ndarray:
    """
    Count unique values in y and return the proportions of each class sorted by label in ascending order.
    Example:
        count(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2])) -> np.array([0.2, 0.3, 0.4, 0.1])
    """
    _, counts = np.unique(y, return_counts=True)

    proportions = counts / counts.sum()

    return proportions


def gini_index(y: np.ndarray) -> float:
    """
    Return the Gini Index of a given NumPy array y.
    The forumla for the Gini Index is 1 - sum(probs^2), where probs are the proportions of each class in y.
    Example:
        gini_index(np.array([1, 1, 2, 2, 3, 3, 4, 4])) -> 0.75
    """

    gini = 1 - np.sum(count(y)**2)

    return gini


def entropy(y: np.ndarray) -> float:
    """
    Return the entropy of a given NumPy array y.
    """
    proportions = count(y)

    entropy = 0
    for el in proportions:
        if el != 0:
            entropy += -el * np.log2(el)

    return entropy


def split(x: np.ndarray, value: float) -> np.ndarray:
    """
    Return a boolean mask for the elements of x satisfying x <= value.
    Example:
        split(np.array([1, 2, 3, 4, 5, 2]), 3) -> np.array([True, True, True, False, False, True])
    """
    return x <= value


def most_common(y: np.ndarray) -> int:
    """
    Return the most common element in y.
    Example:
        most_common(np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])) -> 4
    """
    value, counts = np.unique(y, return_counts=True)

    highest_count_index = np.argmax(counts) # Will return the first highest

    return value[highest_count_index]



def information_gain_entropy(parent: np.ndarray, child1: np.ndarray, child2: np.ndarray) -> float:
    child1weight = len(child1) / len(parent)
    child2weight = len(child2) / len(parent)

    info_gain = entropy(parent) - child1weight * entropy(child1) - child2weight * entropy(child2)
    return info_gain



def find_best_split_mean(X: np.array,y: np.array):

    best_gain = -1
    best_feature = None
    best_threshold = None

    for feature in range(X.shape[1]):
        threshold = np.mean(X[:, feature])


        left_mask = split(X[:, feature], threshold)
        right_mask = ~left_mask

        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            continue


        gain = information_gain_entropy(y, y[left_mask], y[right_mask])

        if gain > best_gain:
            best_gain = gain
            best_feature = feature
            best_threshold = threshold

    return best_feature, best_threshold



class Node:
    """
    A class to represent a node in a decision tree.
    If value != None, then it is a leaf node and predicts that value, otherwise it is an internal node (or root).
    The attribute feature is the index of the feature to split on, threshold is the value to split at,
    and left and right are the left and right child nodes.
    """

    def __init__(
        self,
        feature: int = 0,
        threshold: float = 0.0,
        left: int | Self | None = None,
        right: int | Self | None = None,
        value: int | None = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        # Return True iff the node is a leaf node
        return self.value is not None
    


class DecisionTree:
    def __init__(
        self,
        max_depth: int | None = None,
        criterion: str = "entropy",
    ) -> None:
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth=0,
    ):
        """
        This functions learns a decision tree given (continuous) features X and (integer) labels y.
        """
        # If all the labels are the same, return a pure  leaf node
        if len(np.unique(y)) == 1:
            return Node(value= y[0]) #return a pure leaf node 
        

        # If max depth is not none and the current depth is equal or more than max depth return a leaf node with the most common label
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value = most_common(y))
        


        best_feature, best_threshold = find_best_split_mean(X,y)

        # If it can't find a split return a leaf node with the most common label
        if best_feature == None:
            return Node(value = most_common(y))
        


        left_mask = split(X[:, best_feature], best_threshold)
        right_mask = ~left_mask

        left_subtree = self.fit(X[left_mask], y[left_mask], depth +1)
        right_subtree = self.fit(X[right_mask], y[right_mask], depth +1)


        return Node(feature=best_feature, threshold=best_feature, left=left_subtree, right=right_subtree)
    


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given a NumPy array X of features, return a NumPy array of predicted integer labels.
        """
        for el in X:
            self.root

        ...

if __name__ == "__main__":
    # Test the DecisionTree class on a synthetic dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    seed = 0

    np.random.seed(seed)

    X, y = make_classification(
        n_samples=100, n_features=10, random_state=seed, n_classes=2
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )

    # Expect the training accuracy to be 1.0 when max_depth=None
    rf = DecisionTree(max_depth=None, criterion="entropy")
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")
