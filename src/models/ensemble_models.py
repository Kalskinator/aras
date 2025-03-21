import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import time
from .base_model import BaseModel
from sklearn.preprocessing import LabelEncoder
import logging


class LightGBMModel(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1):
        super().__init__("lightgbm")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.label_encoder = LabelEncoder()

    def train(self, X, y, test_size=0.3, random_state=42):
        logging.info(f"Training LightGBM model...")
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logging.info(
            f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples"
        )

        # Transform labels to continuous integers starting from 0
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        self.model = lgb.LGBMClassifier(
            # n_estimators=self.n_estimators,
            # learning_rate=self.learning_rate,
            random_state=random_state,
            n_jobs=-1,  # Use all available cores
            verbose=1,
        )
        self.model.fit(X_train, y_train_encoded)

        train_time = time.time() - start_time
        logging.info(f"LightGBM training completed in {train_time:.2f} seconds")

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        logging.info(f"\nEvaluating LightGBM model...")
        y_pred_encoded = self.model.predict(X_test)
        # Transform predictions back to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred, zero_division=0)

        if print_report:
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test, y_pred))
            logging.info("\nConfusion Matrix:")
            logging.info(confusion_matrix(y_test, y_pred))

        return accuracy, precision, recall, fscore


class GradientBoostingModel(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__("gradient_boosting")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.label_encoder = LabelEncoder()

    def train(self, X, y, test_size=0.3, random_state=42):
        logging.info(f"Training Gradient Boosting model with {self.n_estimators} estimators...")
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logging.info(
            f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples"
        )

        # Transform labels to continuous integers starting from 0
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        self.model = GradientBoostingClassifier(
            # n_estimators=self.n_estimators,
            # learning_rate=self.learning_rate,
            # max_depth=self.max_depth,
            random_state=random_state,
            verbose=1,
        )
        self.model.fit(X_train, y_train_encoded)

        train_time = time.time() - start_time
        logging.info(f"Gradient Boosting training completed in {train_time:.2f} seconds")

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        logging.info(f"\nEvaluating Gradient Boosting model...")
        y_pred_encoded = self.model.predict(X_test)
        # Transform predictions back to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred, zero_division=0)

        if print_report:
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test, y_pred))
            logging.info("\nConfusion Matrix:")
            logging.info(confusion_matrix(y_test, y_pred))

        return accuracy, precision, recall, fscore


class CatBoostModel(BaseModel):
    def __init__(self, iterations=100, learning_rate=0.1, depth=6):
        super().__init__("catboost")
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.label_encoder = LabelEncoder()

    def train(self, X, y, test_size=0.3, random_state=42):
        logging.info(f"Training CatBoost model...")
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logging.info(
            f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples"
        )

        # Transform labels to continuous integers starting from 0
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        self.model = CatBoostClassifier(
            # iterations=self.iterations,
            # learning_rate=self.learning_rate,
            # depth=self.depth,
            random_seed=random_state,
            thread_count=-1,  # Use all available cores
            verbose=1,
        )
        self.model.fit(X_train, y_train_encoded)

        train_time = time.time() - start_time
        logging.info(f"CatBoost training completed in {train_time:.2f} seconds")

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        logging.info(f"\nEvaluating CatBoost model...")
        y_pred_encoded = self.model.predict(X_test)
        # Transform predictions back to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred, zero_division=0)

        if print_report:
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test, y_pred))
            logging.info("\nConfusion Matrix:")
            logging.info(confusion_matrix(y_test, y_pred))

        return accuracy, precision, recall, fscore


class XGBoostModel(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__("xgboost")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.label_encoder = LabelEncoder()

    def train(self, X, y, test_size=0.3, random_state=42):
        # print(f"Training XGBoost model with {self.n_estimators} estimators...")
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logging.info(
            f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples"
        )

        # Transform labels to continuous integers starting from 0
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        self.model = xgb.XGBClassifier(
            # n_estimators=self.n_estimators,
            # learning_rate=self.learning_rate,
            # max_depth=self.max_depth,
            random_state=random_state,
            n_jobs=-1,  # Use all available cores
        )
        self.model.fit(X_train, y_train_encoded)

        train_time = time.time() - start_time
        logging.info(f"XGBoost training completed in {train_time:.2f} seconds")

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        logging.info(f"\nEvaluating XGBoost model...")
        y_pred_encoded = self.model.predict(X_test)
        # Transform predictions back to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred)

        if print_report:
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test, y_pred))
            logging.info("\nConfusion Matrix:")
            logging.info(confusion_matrix(y_test, y_pred))

        return accuracy, precision, recall, fscore
