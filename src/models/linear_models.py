from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import time
from .base_model import BaseModel
import numpy as np


class SupportVectorMachineModel(BaseModel):
    def __init__(self, C=100, kernel="rbf", gamma=0.001):
        super().__init__("svm")
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

    def train(self, X, y, test_size=0.3, random_state=42):
        print(f"Training SVM model with Gamma={self.gamma} and C={self.C}...")
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")

        self.model = SVC(
            C=self.C,
            # kernel=self.kernel,
            gamma=self.gamma,
            probability=True,
            random_state=random_state,
            verbose=True,
        )
        self.model.fit(X_train, y_train)

        train_time = time.time() - start_time
        print(f"SVM training completed in {train_time:.2f} seconds")

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        print(f"\nEvaluating SVM model...")
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred)

        if print_report:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

        return accuracy, precision, recall, fscore


class LogisticRegressionModel(BaseModel):
    def __init__(self, C=1.0, solver="lbfgs"):
        super().__init__("logistic_regression")
        self.C = C
        self.solver = solver

    def train(self, X, y, test_size=0.3, random_state=42):
        # print(f"Training Logistic Regression model with solver={self.solver}, C={self.C}...")
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")

        self.model = LogisticRegression(
            # C=self.C,
            # solver=self.solver,
            # max_iter=1000,
            # multi_class="auto",
            # random_state=random_state,
        )
        self.model.fit(X_train, y_train)

        train_time = time.time() - start_time
        print(f"Logistic Regression training completed in {train_time:.2f} seconds")

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        print(f"\nEvaluating Logistic Regression model...")
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred)

        if print_report:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

        return accuracy, precision, recall, fscore
