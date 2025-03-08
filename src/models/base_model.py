from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def train(self, X, y, test_size=0.3, random_state=42):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test, print_report=False):
        pass
