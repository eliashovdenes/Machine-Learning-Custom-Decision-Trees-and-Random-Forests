import numpy as np

from decision_tree import DecisionTree


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: None | int = 5,
        criterion: str = "entropy",
        max_features: None | str = "sqrt",
        seed: int = 0
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.trees = []
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray):

        np.random.seed(self.seed) # set seed

        num_trees = self.n_estimators   # num of trees to be created

        n = len(X) # how many features we have
   
        for _ in range(num_trees): # for each tree

            X_indexes = np.random.choice(np.arange(n), size=n, replace=True) #choose features at random, where features can occur multiple times

            X_sampled = X[X_indexes] 

            y_sampled = y[X_indexes]

            rf = DecisionTree(max_depth=self.max_depth, criterion= self.criterion, max_features=self.max_features) # create the tree
            rf.fit(X_sampled, y_sampled) # fit the tree
            self.trees.append(rf) # add the tree to the list


     

    def predict(self, X: np.ndarray) -> np.ndarray:

        list_of_predicts = []

        for index in range(len(self.trees)): #for every tree
            predict = self.trees[index].predict(X) #get the current tree and predict all the X values for that tree
            list_of_predicts.append(predict) # add to list

        list_of_predicts = np.array(list_of_predicts) #convert into np.array

        # compare every prediction
        prediction = []

        for index in range(len(X)): # for each feature in X
            labels = list_of_predicts[:, index] # this becomes a list of all the trees prediction of the current feature 

            most_common_element = np.bincount(labels).argmax() # get the most common target for the current feature
            prediction.append(most_common_element) # add this to the list
        

        return prediction

if __name__ == "__main__":
    # Test the RandomForest class on a synthetic dataset
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

    rf = RandomForest(
        n_estimators=20, max_depth=None, criterion="entropy", max_features="sqrt"
    )
    rf.fit(X_train, y_train)

    

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")
