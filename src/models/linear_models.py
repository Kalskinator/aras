from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import time
from .base_model import BaseModel
import numpy as np
from src.utils.progress_bar_helper import ProgressBarHelper
import logging


# Maybe rename to SGDClassifierModel
class SupportVectorMachineModel(BaseModel):
    def __init__(self, C=100, kernel="rbf", gamma=0.001):
        super().__init__("svm")
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

    def train(self, X, y, test_size=0.3, random_state=42, progress_bar=None):
        logging.info(f"Training SVM model with Gamma={self.gamma} and C={self.C}...")
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logging.info(
            f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples"
        )

        self.model = SGDClassifier(
            loss="hinge",
            max_iter=1000,
            verbose=0,  # Set to 0 to disable built-in verbosity when using our progress bar
            tol=1e-3,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
        )

        # Use progress bar for training
        if progress_bar:
            progress_bar_helper = ProgressBarHelper(total=4, desc="Training SVM")

            # SGD is iterative, so we can update the progress bar as it trains
            self.model.fit(X_train, y_train)
            for i in range(4):
                progress_bar_helper.update(1)
            progress_bar_helper.close()
        else:
            self.model.fit(X_train, y_train)

        train_time = time.time() - start_time
        logging.info(f"SVM training completed in {train_time:.2f} seconds")

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        logging.info(f"\nEvaluating SVM model...")
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred)

        if print_report:
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test, y_pred))
            logging.info("\nConfusion Matrix:")
            logging.info(confusion_matrix(y_test, y_pred))

        return accuracy, precision, recall, fscore


class LogisticRegressionModel(BaseModel):
    def __init__(self, C=1.0, solver="lbfgs"):
        super().__init__("logistic_regression")
        self.C = C
        self.solver = solver

    def train(self, X, y, test_size=0.3, random_state=42):
        # logging.info(
        #     f"Training Logistic Regression model with solver={self.solver}, C={self.C}..."
        # )
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logging.info(
            f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples"
        )

        self.model = LogisticRegression(
            # C=self.C,
            # solver=self.solver,
            # max_iter=1000,
            # multi_class="auto",
            # random_state=random_state,
        )
        self.model.fit(X_train, y_train)

        train_time = time.time() - start_time
        logging.info(f"Logistic Regression training completed in {train_time:.2f} seconds")

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        logging.info(f"\nEvaluating Logistic Regression model...")
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred)

        if print_report:
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test, y_pred))
            logging.info("\nConfusion Matrix:")
            logging.info(confusion_matrix(y_test, y_pred))

        return accuracy, precision, recall, fscore
