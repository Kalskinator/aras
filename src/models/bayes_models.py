from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import time
from .base_model import BaseModel
import logging


class GaussianNBModel(BaseModel):
    def __init__(self):
        super().__init__("gaussiannb")

    def train(self, X, y, test_size=0.3, random_state=42):
        logging.info("Training Gaussian Naive Bayes model...")
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logging.info(
            f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples"
        )

        self.model = GaussianNB()
        self.model.fit(X_train, y_train)

        train_time = time.time() - start_time
        logging.info(f"Gaussian Naive Bayes training completed in {train_time:.2f} seconds")

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        logging.info("\nEvaluating Gaussian Naive Bayes model...")
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred, zero_division=0)

        if print_report:
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test, y_pred))
            logging.info("\nConfusion Matrix:")
            logging.info(confusion_matrix(y_test, y_pred))

        return accuracy, precision, recall, fscore
