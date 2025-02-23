from sklearn.ensemble import RandomForestClassifier
from src.Data.data_loader import load_all_data
from src.models.decision_tree import train_decision_tree, evaluate_model
from src.config import SENSOR_COLUMNS_HOUSE_A, ACTIVITY_COLUMNS, DATA_DIR_HOUSE_A
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import joblib  # For saving and loading the model

# Read the data
# data_file = "House A/DAY_1.csv"  # Using House A data
# df = pd.read_csv(data_file)

# data_file = "House A/DAY_1.txt"  # Using House A data
# df = pd.read_csv(data_file, sep="\s+", header=None)
# df.columns = sensor_columns + activity_columns
# X = df.iloc[:, :20]  # First 20 columns are sensor data
# y = df.iloc[:, 20]  # Column 21 is resident 1's activities


print(f"Loading data from {DATA_DIR_HOUSE_A}")
df = load_all_data(DATA_DIR_HOUSE_A)

# GOOOD METHOD
feature_columns = ["Time"] + SENSOR_COLUMNS_HOUSE_A

X = df[feature_columns]  # Use only sensor columns
y = df["Activity_R2"]  # Predict activities for resident 1

# CHEATING METHOD
# feature_columns = ["Time"] + SENSOR_COLUMNS_HOUSE_A + ACTIVITY_COLUMNS

# y = df["Activity_R1"]
# df.drop(["Activity_R1"], axis=1, inplace=True)
# X = df

# print("\nDataset information:")
# print(f"Total number of samples: {len(df)}")
# print(f"Number of days: {df['Day'].nunique()}")
# print("\nSamples per day:")
# print(df.groupby("Day").size())

# # Train and evaluate the model
model, X_train, X_test, y_train, y_test = train_decision_tree(X, y)
evaluate_model(model, X_test, y_test)

# Save the model using joblib
joblib.dump(model, "model.joblib")
print("Model saved as 'model.joblib'")

# Initialize the SHAP JavaScript library
shap.initjs()

# Load the model from the saved file
loaded_model = joblib.load("model.joblib")
print("Model loaded from 'model.joblib'")

# SHAP Explainer
explainer = shap.TreeExplainer(model, X_train)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, feature_names=SENSOR_COLUMNS_HOUSE_A, plot_type="bar")
plt.savefig('shap_summary_plot.png')  # Save the plot to a file
print("SHAP summary plot saved as 'shap_summary_plot.png'")

# SHAP stacked force plot
force_plot = shap.force_plot(explainer.expected_value, shap_values, X_test, feature_names=SENSOR_COLUMNS_HOUSE_A)
shap.save_html('shap_force_plot.html', force_plot)  # Save the plot to an HTML file
print("SHAP force plot saved as 'shap_force_plot.html'")