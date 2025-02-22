import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Read the data
data_file = "House A/DAY_1.txt"
df = pd.read_csv(data_file, sep="\s+", header=None)

# Separate features (sensors) and target (activity labels for resident 1)
X = df.iloc[:, :20]  # First 20 columns are sensor data
y = df.iloc[:, 20]  # Column 21 is resident 1's activities

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Print performance metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print model accuracy
accuracy = dt_classifier.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy:.2f}")
