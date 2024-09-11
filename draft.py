# Imports
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# ... add more imports as needed

# My implementations
from decision_tree import DecisionTree
from random_forest import RandomForest


""" Count unique values in y and return the proportions of each class sorted by label in ascending order.
    Example:
        count(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2])) -> np.array([0.2, 0.3, 0.4, 0.1])
    """

y = np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2])

x = np.bincount(y)

# # x, kuk = np.unique(y, return_counts=True)
# list = []
# for el in kuk:
#     senip = el/len(y)
#     list.append(senip)

print(x)


x = x / x.sum()

print(np.array(listen))






