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

    _, counts = np.unique(y, return_counts=True)

    proportions = counts / counts.sum()

    return proportions


# x = count(np.array([1, 1, 2, 2, 3, 3, 4, 4]))

# print(x)






def gini_index(y: np.ndarray) -> float:
    """
    Return the Gini Index of a given NumPy array y.
    The forumla for the Gini Index is 1 - sum(probs^2), where probs are the proportions of each class in y.
    Example:
        gini_index(np.array([1, 1, 2, 2, 3, 3, 4, 4])) -> 0.75
    """
    
    gini = 1 - np.sum(count(y)**2)

    return gini


# x = gini_index(np.array([1, 1, 2, 2, 3, 3, 4, 4, 0]))

# print(x)



def entropy(y: np.ndarray) -> float:
    """
    Return the entropy of a given NumPy array y.
    """
    proportions = count(y)

    print(proportions)

    entropy = 0
    for el in proportions:
        if el != 0:
            entropy += -el * np.log2(el)

    return entropy


x = entropy(np.array([1, 1, 2, 2, 3, 3, 4, 4]))

print(x)

y = entropy(np.array([2, 2, 3, 3]))

print(y)

z = entropy(np.array([3, 3, 3, 3]))

print(z)



