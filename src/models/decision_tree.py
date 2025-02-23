from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def train_decision_tree(X, y, test_size=0.2, random_state=42):
    """Train a decision tree classifier."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Create and train the model
    dt_classifier = DecisionTreeClassifier(criterion="gini")
    dt_classifier.fit(X_train, y_train)
    print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", dt_classifier.score(X_test, y_test))

    return dt_classifier, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    # plt.title("Confusion Matrix")
    # plt.ylabel("True Label")
    # plt.xlabel("Predicted Label")
    # plt.show()

    accuracy = model.score(X_test, y_test)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    precision, recall, fscore, support = score(y_test, y_pred)

    print(f"Precision: {np.mean(precision):.2f}")
    print(f"Recall: {np.mean(recall):.2f}")
    print(f"Fscore: {np.mean(fscore):.2f}")
