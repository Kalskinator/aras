from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import time
from tqdm import tqdm
from .base_model import BaseModel
import numpy as np
import matplotlib.pyplot as plt
import logging


class DecisionTreeModel(BaseModel):
    def __init__(self):
        super().__init__("decision_tree")

    def train(self, X, y, test_size=0.3, random_state=42):
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logging.info(
            f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples"
        )

        self.model = DecisionTreeClassifier(random_state=random_state, criterion="gini")

        self.model.fit(X_train, y_train)

        train_time = time.time() - start_time
        logging.info(f"Decision Tree training completed in {train_time:.2f} seconds")
        logging.info(f"Decision Tree Accuracy: {self.model.score(X_test, y_test):.4f}")

        # importances = self.model.feature_importances_
        # feature_names = X_train.columns

        # indices = np.argsort(importances)[::-1]

        # # Plot
        # plt.figure(figsize=(10, 5))
        # plt.title("Feature Importance Decision Tree Model (Gini)")
        # plt.bar(range(len(importances)), importances[indices], align="center")
        # plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        # plt.show()

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        logging.info(f"\nEvaluating Decision Tree model...")
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred)

        if print_report:
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test, y_pred))
            logging.info("\nConfusion Matrix:")
            logging.info(confusion_matrix(y_test, y_pred))

        return accuracy, precision, recall, fscore


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__("random_forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def train(self, X, y, test_size=0.3, random_state=42):
        start_time = time.time()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logging.info(
            f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples"
        )

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=random_state,
            n_jobs=-1,  # Use all available cores
            warm_start=True,  # Enable warm start to fit incrementally
        )

        self.model.fit(X_train, y_train)

        train_time = time.time() - start_time

        # logging.info(f"Random Forest training completed in {train_time:.2f} seconds")
        # logging.info(f"Random Forest Accuracy: {self.model.score(X_test, y_test):.4f}")

        # importances = self.model.feature_importances_
        # feature_names = X_train.columns

        # indices = np.argsort(importances)[::-1]

        # plt.figure(figsize=(10, 5))
        # plt.title("Feature Importance Random Forest Model (Gini)")
        # plt.bar(range(len(importances)), importances[indices], align="center")
        # plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        # plt.show()

        logging.info(f"Random Forest training completed in {train_time:.2f} seconds")
        logging.info(f"Random Forest Accuracy: {self.model.score(X_test, y_test):.4f}")

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        logging.info(f"\nEvaluating Random Forest model...")
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred)

        if print_report:
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test, y_pred))
            logging.info("\nConfusion Matrix:")
            logging.info(confusion_matrix(y_test, y_pred))

        return accuracy, precision, recall, fscore
