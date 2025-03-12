from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import time
from .base_model import BaseModel
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import logging


class SupportVectorMachineModel(BaseModel):
    def __init__(self, C=100, kernel="rbf", gamma=0.001):
        super().__init__("svm")
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

    def train(self, X, y, test_size=0.3, random_state=42):
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
            verbose=0,
            tol=1e-3,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
        )

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


class PolynomialSVMModel(BaseModel):
    def __init__(self, C=1.0, degree=2):
        super().__init__("poly_svm")
        self.C = C
        self.degree = degree

    def train(self, X, y, test_size=0.3, random_state=42):
        logging.info(f"Training Polynomial SVM model with degree={self.degree}...")
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logging.info(
            f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples"
        )

        # Create a pipeline with preprocessing and model
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=self.degree, include_bias=False)),
                (
                    "svm",
                    SGDClassifier(
                        loss="hinge",
                        alpha=1 / (self.C * X_train.shape[0]),  # Alpha is 1/(C*n_samples)
                        max_iter=1000,
                        tol=1e-3,
                        random_state=random_state,
                        early_stopping=True,
                    ),
                ),
            ]
        )

        # Train the model
        self.model.fit(X_train, y_train)

        train_time = time.time() - start_time
        logging.info(f"Polynomial SVM training completed in {train_time:.2f} seconds")

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        """Evaluate the model on test data."""
        logging.info(f"\nEvaluating Polynomial SVM model...")
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
        logging.info(f"Training Logistic Regression model...")
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
