import numpy as np

from decision_tree import DecisionTree


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: None | int = 5,
        criterion: str = "entropy",
        max_features: None | str = "sqrt",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):

        trees = self.n_estimators   

        n = len(X)

        X_indexes = np.random.choice(np.arange(n), size=n, replace=True)

        X_sampled = X[X_indexes]

        y_sampled = y[X_indexes]

   
        for _ in range(trees):
            rf = DecisionTree(max_depth=self.max_depth, criterion= self.criterion, max_features=self.max_features)
            rf.fit(X_sampled, y_sampled)
            self.trees.append(rf)


     

    def predict(self, X: np.ndarray) -> np.ndarray:
        

        list_of_predicts = []
        for index in range(len(self.trees)):
            predict = self.trees[index].predict(X)
            list_of_predicts.append(predict)

        list_of_predicts = np.array(list_of_predicts)

        # compare every prediction

        prediction = []

        for index in range(len(X)):
            labels = list_of_predicts[:, index]

            most_common_element = np.bincount(labels).argmax()
            prediction.append(most_common_element)
        

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
