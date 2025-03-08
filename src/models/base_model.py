from abc import ABC, abstractmethod
import os
import pickle
import joblib
from datetime import datetime


class BaseModel(ABC):
    def __init__(self, name):
        self.name = name
        self.model = None

    @abstractmethod
    def train(self, X, y, test_size=0.3, random_state=42):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test, print_report=False):
        pass

    def save(self, artifacts_dir="artifacts/models", accuracy=None):
        """Save the trained model to disk"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # # Create directory for this model type if it doesn't exist
        # model_dir = os.path.join(artifacts_dir, self.name)
        # os.makedirs(model_dir, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{timestamp}_ACCURACY_{accuracy:.2f}.pkl"
        filepath = os.path.join(artifacts_dir, filename)

        # Save the model
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
        return filepath

    def load(self, filepath):
        """Load a trained model from disk"""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self
