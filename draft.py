# Imports
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# ... add more imports as needed

# My implementations
from decision_tree import DecisionTree
from random_forest import RandomForest


def count(y: np.ndarray) -> np.ndarray:
    """
    Count unique values in y and return the proportions of each class sorted by label in ascending order.
    Example:
        count(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2])) -> np.array([0.2, 0.3, 0.4, 0.1])
    """

    counts  = np.bincount(y)

    proportions = counts / counts.sum()

    return proportions


# x = count(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2]))

# print(x)

# print(type(x))


def gini_index(y: np.ndarray) -> float:
    """
    Return the Gini Index of a given NumPy array y.
    The forumla for the Gini Index is 1 - sum(probs^2), where probs are the proportions of each class in y.
    Example:
        gini_index(np.array([1, 1, 2, 2, 3, 3, 4, 4])) -> 0.75
    """
    
    gini = 1 - np.sum(count(y)**2)

    return gini


x = gini_index(np.array([1, 1, 2, 2, 3, 3, 4, 4]))

print(x)



