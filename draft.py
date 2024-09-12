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

    

    entropy = 0
    for el in proportions:
        if el != 0:
            entropy += -el * np.log2(el)

    return entropy


# x = entropy(np.array([1, 1, 2, 2, 3, 3, 4, 4]))

# print(x)



def split(x: np.ndarray, value: float) -> np.ndarray:
    """
    Return a boolean mask for the elements of x satisfying x <= value.
    Example:
        split(np.array([1, 2, 3, 4, 5, 2]), 3) -> np.array([True, True, True, False, False, True])
    """

    

    return x <= value


# x = split(np.array([1, 2, 3, 4, 5, 2]), 3)

# print(x)

def most_common(y: np.ndarray) -> int:
    """
    Return the most common element in y.
    Example:
        most_common(np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])) -> 4
    """

    value, counts = np.unique(y, return_counts=True)

    highest_count_index = np.argmax(counts) # Will return the first highest

    return value[highest_count_index]




# x = most_common(np.array([1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]))

# print(x)



def information_gain(parent: np.ndarray, child1: np.ndarray, child2: np.ndarray) -> float:
    child1weight = len(child1) / len(parent)
    child2weight = len(child2) / len(parent)

    info_gain = entropy(parent) - child1weight * entropy(child1) - child2weight * entropy(child2)
    return info_gain



parent = np.array([1,1,1,1])
child1 = np.array([1,1,1])

child2 = np.array([1,1,1,1])


x = information_gain(parent, child1, child2)

# print(x)




# x = len(np.unique([0,0,0,0,0]))

# print(x)


X = np.array([[2,2,4,5],[3,2,2,1]])
for feature in range(X.shape[1]):
        # print(feature)
        # print(np.mean(X[:, feature]))
        threshold = np.mean(X[:, feature])
        # print(X[:,feature])

        # print(threshold)




        left_mask = split(X[:, feature], threshold)
        right_mask = ~left_mask


        # print(left_mask)
        # print(right_mask)

        # print(type(left_mask[0]))



X = np.array([[1,1,1,2],[2,2,3,3]])


if True:
    # Calculate n as the square root of the number of features
    n = int(np.sqrt(X.shape[1]))  # Get the number of features to select

    # Randomly select n indices from the range of available features (0 to X.shape[1] - 1)
    selected_features = np.random.choice(np.arange(X.shape[1]), size=n, replace=False)

    print("Selected", selected_features)


print(np.arange(X.shape[1]))



for x in range(20):
    print(x)