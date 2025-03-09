from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import time
from .base_model import BaseModel
import numpy as np


class KNearestNeighborsModel(BaseModel):
    def __init__(self, n_neighbors=5, metric="manhattan"):
        super().__init__("knn")
        self.n_neighbors = n_neighbors
        self.metric = metric

    def train(self, X, y, test_size=0.3, random_state=42):
        print(f"Training KNN model with {self.n_neighbors} neighbors...")
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")

        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors, metric=self.metric, n_jobs=-1  # Use all available cores
        )
        self.model.fit(X_train, y_train)

        train_time = time.time() - start_time
        print(f"KNN training completed in {train_time:.2f} seconds")

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        print(f"\nEvaluating KNN model...")
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred)

        if print_report:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

        return accuracy, precision, recall, fscore
