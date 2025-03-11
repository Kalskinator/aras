import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import time
from .base_model import BaseModel
from sklearn.preprocessing import LabelEncoder


#Best parameters found: {'subsample': 1.0, 'num_leaves': 100, 'n_estimators': 50, 'min_child_samples': 100, 'max_depth': -1, 'learning_rate': 0.1, 'colsample_bytree': 0.6}
#Best cross-validation score: 0.8448
#Best parameters for lightgbm: {'subsample': 1.0, 'num_leaves': 100, 'n_estimators': 50, 'min_child_samples': 100, 'max_depth': -1, 'learning_rate': 0.1, 'colsample_bytree': 0.6}
#Best cross-validation score for lightgbm: 0.8448
class LightGBMModel(BaseModel):
    def __init__(self):
        super().__init__("lightgbm")
        self.n_estimators = 50
        self.learning_rate = 0.1
        self.num_leaves = 100
        self.max_depth = -1
        self.min_child_samples = 100
        self.subsample = 1.0
        self.colsample_bytree = 0.6
        self.label_encoder = LabelEncoder()

    def train(self, X, y, test_size=0.3, random_state=42):
        # print(f"Training LightGBM model with {self.n_estimators} estimators...")
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")

        # Transform labels to continuous integers starting from 0
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        self.model = lgb.LGBMClassifier(
<<<<<<< Updated upstream
            # n_estimators=self.n_estimators,
            # learning_rate=self.learning_rate,
            random_state=random_state,
            n_jobs=-1,  # Use all available cores
        )
        self.model.fit(X_train, y_train_encoded)
=======
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=random_state,
            n_jobs=-1,  # Use all available cores
        )

        # Define a custom callback to update the progress bar
        if progress_bar:
            progress_bar_helper = ProgressBarHelper(total=self.n_estimators // 4, desc="Training LightGBM")  # 25% increments
            def callback(env):
                progress_bar_helper.update(1)

            self.model.fit(
                X_train, y_train_encoded,
                eval_set=[(X_test, y_test_encoded)],
                callbacks=[callback]
            )
            progress_bar_helper.close()
        else:
            self.model.fit(X_train, y_train_encoded)
>>>>>>> Stashed changes

        train_time = time.time() - start_time
        print(f"LightGBM training completed in {train_time:.2f} seconds")

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        print(f"\nEvaluating LightGBM model...")
        y_pred_encoded = self.model.predict(X_test)
        # Transform predictions back to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred, zero_division=0)

        if print_report:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

        return accuracy, precision, recall, fscore

    ## Hyperparameter tuning function

    # def tune_hyperparameters(self, X, y, param_grid, n_iter=25, cv=3, scoring='accuracy'):
    #     """Tune hyperparameters using RandomizedSearchCV."""
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=0.3, random_state=42, stratify=y
    #     )
    #     print(f"Tuning hyperparameters on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")

    #     # Fit the label encoder on both training and validation labels
    #     self.label_encoder.fit(y)
    #     y_train_encoded = self.label_encoder.transform(y_train)

    #     random_search = RandomizedSearchCV(
    #         estimator=lgb.LGBMClassifier(random_state=42, n_jobs=-1),
    #         param_distributions=param_grid,
    #         n_iter=n_iter,
    #         scoring=scoring,
    #         cv=cv,
    #         verbose=1,
    #         n_jobs=-1
    #     )

    #     random_search.fit(X_train, y_train_encoded)

    #     print(f"Best parameters found: {random_search.best_params_}")
    #     print(f"Best cross-validation score: {random_search.best_score_:.4f}")

    #     self.model = random_search.best_estimator_

    #     return random_search.best_params_, random_search.best_score_


class GradientBoostingModel(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__("gradient_boosting")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.label_encoder = LabelEncoder()

    def train(self, X, y, test_size=0.3, random_state=42):
        # print(f"Training Gradient Boosting model with {self.n_estimators} estimators...")
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")

        # Transform labels to continuous integers starting from 0
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        self.model = GradientBoostingClassifier(
            # n_estimators=self.n_estimators,
            # learning_rate=self.learning_rate,
            # max_depth=self.max_depth,
            random_state=random_state,
        )
        self.model.fit(X_train, y_train_encoded)

        train_time = time.time() - start_time
        print(f"Gradient Boosting training completed in {train_time:.2f} seconds")

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        print(f"\nEvaluating Gradient Boosting model...")
        y_pred_encoded = self.model.predict(X_test)
        # Transform predictions back to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred, zero_division=0)

        if print_report:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

        return accuracy, precision, recall, fscore


class CatBoostModel(BaseModel):
    def __init__(self, iterations=100, learning_rate=0.1, depth=6):
        super().__init__("catboost")
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.label_encoder = LabelEncoder()

    def train(self, X, y, test_size=0.3, random_state=42):
        # print(f"Training CatBoost model with {self.iterations} iterations...")
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")

        # Transform labels to continuous integers starting from 0
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        self.model = CatBoostClassifier(
            # iterations=self.iterations,
            # learning_rate=self.learning_rate,
            # depth=self.depth,
            random_seed=random_state,
            verbose=1,
            thread_count=-1,  # Use all available cores
        )
        self.model.fit(X_train, y_train_encoded)

        train_time = time.time() - start_time
        print(f"CatBoost training completed in {train_time:.2f} seconds")

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        print(f"\nEvaluating CatBoost model...")
        y_pred_encoded = self.model.predict(X_test)
        # Transform predictions back to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred, zero_division=0)

        if print_report:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

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
        print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")

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
        print(f"XGBoost training completed in {train_time:.2f} seconds")

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test, print_report=False):
        print(f"\nEvaluating XGBoost model...")
        y_pred_encoded = self.model.predict(X_test)
        # Transform predictions back to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred)

        if print_report:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

        return accuracy, precision, recall, fscore
