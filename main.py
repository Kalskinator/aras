import numpy as np
import pandas as pd
import shap
import seaborn as sns
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

sensor_columns = [
    "Ph1_Wardrobe",
    "Ph2_ConvertibleCouch",
    "Ir1_TV",
    "Fo1_Couch",
    "Fo2_Couch",
    "Di3_Chair",
    "Di4_Chair",
    "Ph3_Fridge",
    "Ph4_KitchenDrawer",
    "Ph5_Wardrobe",
    "Ph6_BathroomCabinet",
    "Co1_HouseDoor",
    "Co2_BathroomDoor",
    "Co3_ShowerDoor",
    "So1_Hall",
    "So2_Kitchen",
    "Di1_Tap",
    "Di2_WaterCloset",
    "Te1_Kitchen",
    "Fo3_Bed",
]
activity_columns = ["Activity_R1", "Activity_R2"]


def load_day_data(file_path):
    df = pd.read_csv(file_path, sep="\s+", header=None, names=sensor_columns + activity_columns)
    return df


all_data = []
data_dir = Path("House A")

# Read the data
# data_file = "House A/DAY_1.csv"  # Using House A data
# df = pd.read_csv(data_file)

# data_file = "House A/DAY_1.txt"  # Using House A data
# df = pd.read_csv(
#     data_file, delim_whitespace=True, header=None  # Use whitespace as delimiter
# )  # No header row in the file
# df.columns = sensor_columns + activity_columns
# X = df.iloc[:, :20]  # First 20 columns are sensor data
# y = df.iloc[:, 20]  # Column 21 is resident 1's activities

# Loop through all DAY_*.txt files
for day_file in sorted(data_dir.glob("DAY_*.txt"), key=lambda x: int(x.stem.split("_")[1])):
    print(f"Loading {day_file}")
    day_data = load_day_data(day_file)
    day_data["Day"] = day_file.stem.split("_")[1]
    all_data.append(day_data)

df = pd.concat(all_data, ignore_index=True)

# print("\nDataset information:")
# print(f"Total number of samples: {len(df)}")
# print(f"Number of days: {df['Day'].nunique()}")
# print("\nSamples per day:")
# print(df.groupby("Day").size())

X = df[sensor_columns]  # Use only sensor columns
y = df["Activity_R2"]  # Predict activities for resident 1

print(X.head())
print(y.head())


# # Split the data using stratified sampling to ensure all classes are represented
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Create and train the decision tree
# dt_classifier = DecisionTreeClassifier(criterion="entropy")
# dt_classifier.fit(X_train, y_train)

# # Print class distribution
# print("\nClass distribution in training set:")
# print(pd.Series(y_train).value_counts())

# print("\nClass distribution in test set:")
# print(pd.Series(y_test).value_counts())

# # Make predictions
# y_pred = dt_classifier.predict(X_test)

# # Print performance metrics
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))


# accuracy = dt_classifier.score(X_test, y_test)
# print(f"\nModel Accuracy: {accuracy:.2f}")
