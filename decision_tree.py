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



def information_gain(parent: np.ndarray, child1: np.ndarray, child2: np.ndarray, criterion: str) -> float:

    child1weight = len(child1) / len(parent) #calculate weight of child1
    child2weight = len(child2) / len(parent) #calculate weight of child2

    ## Calculate the impurity of the parent node
    if criterion == "gini":
        parent_impurity = gini_index(parent)
        child1_impurity = gini_index(child1)
        child2_impurity = gini_index(child2)
    elif criterion == "entropy":
        parent_impurity = entropy(parent)
        child1_impurity = entropy(child1)
        child2_impurity = entropy(child2)

    else:
        print("Criterion not set correctly:", criterion)

    

    # Calculate the information gain
    info_gain = parent_impurity - (child1weight * child1_impurity + child2weight * child2_impurity)


    return info_gain

    

    




def find_best_split_mean(X: np.array,y: np.array, self):

    np.random.seed(self.seed) # Sets the seed as set in the init of decision tree

    best_gain = -1
    best_feature = None
    best_threshold = None
    selected_features = None

    # Consider all features
    if self.max_features is None:
        selected_features = np.arange(X.shape[1])

    # Consider the square root of features
    elif self.max_features == "sqrt":
        n = int(np.sqrt(X.shape[1])) 

        selected_features = np.random.choice(np.arange(X.shape[1]), size=n, replace=False)

    # Conisider the log2 of features
    elif self.max_features == "log2":
        n = int(np.log2(X.shape[1])) 

        selected_features = np.random.choice(np.arange(X.shape[1]), size=n, replace=False)
    else:
        raise ValueError("Invalid value for max_features")

    # Go through the features and find the best split
    for feature in selected_features:
        threshold = np.mean(X[:, feature]) # Compute the threshold


        left_mask = split(X[:, feature], threshold) # Mask with the left part of the split
        right_mask = ~left_mask # Mask with the right part of the split

        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0: # If the split is invalid continue to try to find a split
            continue


        gain = information_gain(y, y[left_mask], y[right_mask], self.criterion) # compute the information gain of the split

        if gain > best_gain: # if the gain is better than default or the previous then choose that split
            best_gain = gain
            best_feature = feature
            best_threshold = threshold

    return best_feature, best_threshold


def print_tree(node, level=0):
    if node.is_leaf():
        print("  " * level + f"Leaf: Predict {node.value}")
    else:
        print("  " * level + f"Node: Feature {node.feature} <= {node.threshold}")
        if node.left:
            print_tree(node.left, level + 1)
        if node.right:
            print_tree(node.right, level + 1)



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
    



def predict_recursive(currentNode: Node, X: np.ndarray) -> int:

    if currentNode.is_leaf():
        return currentNode.value # If leaf return the value of leaf
    
    feature = currentNode.feature

    threshold = currentNode.threshold

    # decide to go to left or right node
    if X[feature] <= threshold:
        return predict_recursive(currentNode.left, X)
    else:
        return predict_recursive(currentNode.right, X)
    


class DecisionTree:
    def __init__(
        self,
        max_depth: int | None = None,
        criterion: str = "entropy",
        max_features: str | None = None,
        seed: int = 0
    ) -> None:
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.seed = seed
        if (max_depth is not None and max_depth <= 1):
            raise Exception("The tree can not be less then 2 in depth")
        

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth=0,
    ):
        """
        This functions learns a decision tree given (continuous) features X and (integer) labels y.
        """
        
        # If the depth is at 0 we will assign the root and continue the creation of the tree
        if depth == 0:            
            self.root = self.fit(X, y, depth = 1)
            return self.root
        
        
        # If all the labels are the same, return a pure  leaf node
        if len(np.unique(y)) == 1:
            return Node(value= y[0]) #return a pure leaf node 
        

        # If max depth is not none and the current depth is equal or more than max depth return a leaf node with the most common label
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value = most_common(y))
        

        # Return the best feature and the threshold where we split using the find_best_split_mean function
        best_feature, best_threshold = find_best_split_mean(X,y, self)

        # If it can't find a split return a leaf node with the most common label
        if best_feature == None:
            return Node(value = most_common(y))
        

        # Here we split the X values on the feature we found this will return lists with true and false
        left_mask = split(X[:, best_feature], best_threshold)
        right_mask = ~left_mask  # Opposite list with true and false then above


        # Create the two subtrees
        left_subtree = self.fit(X[left_mask], y[left_mask], depth +1)
        right_subtree = self.fit(X[right_mask], y[right_mask], depth +1)


        # Keep splitting the values and return the two subtrees as left and right in the node
        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)
    

    
    
    

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given a NumPy array X of features, return a NumPy array of predicted integer labels.
        """
        list_of_predicts = []

        threshold = self.root.threshold

        feature = self.root.feature

        for el in X: # for each feature 
            xfeatureval = el[feature] # get the feature value


            # decide if we are going to the left or right depending on the threshold and the feature value
            if xfeatureval <= threshold:
                list_of_predicts.append(predict_recursive(self.root.left, el))

            else:
                list_of_predicts.append(predict_recursive(self.root.right, el))


        return np.array(list_of_predicts)
            

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

    
    rf = DecisionTree(max_depth=2, criterion="entropy", max_features="sqrt")
    rf.fit(X_train, y_train)

    print_tree(rf.root)


    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")
