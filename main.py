from sklearn.ensemble import RandomForestClassifier
from src.Data.data_loader import load_all_data
from src.models.decision_tree import train_decision_tree, evaluate_model
from src.config import SENSOR_COLUMNS_HOUSE_A, ACTIVITY_COLUMNS, DATA_DIR_HOUSE_A, ACTIVITY_MAPPING
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import joblib  # For saving and loading the model
import signal  # For setting a timeout
import numpy as np  # For shape checking

# Set Matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Function to handle timeout
def handler(signum, frame):
    raise TimeoutError("SHAP calculation timed out")

# Set the timeout signal
signal.signal(signal.SIGALRM, handler)
signal.alarm(300)  # Set timeout to 300 seconds (5 minutes)

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

# Reduce the size of the test set for SHAP calculation
X_test_sample = X_test.sample(n=100, random_state=42)
print(f"Reduced test set size: {X_test_sample.shape}")

# SHAP Explainer
print("Initializing SHAP explainer")
explainer = shap.TreeExplainer(loaded_model)
print("Calculating SHAP values")
try:
    shap_values = explainer.shap_values(X_test_sample)
    print("SHAP values calculated")
    print(f"Shape of SHAP values: {np.array(shap_values).shape}")
    print(f"Shape of X_test_sample: {X_test_sample.shape}")
except TimeoutError:
    print("SHAP calculation timed out")
    shap_values = None
""" 
# Generate and display SHAP force plot if shap_values is not None
if shap_values is not None:
    force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1], X_test_sample)
    shap.save_html('shap_force_plot.html', force_plot)  # Save the plot to an HTML file
    print("SHAP force plot saved as 'shap_force_plot.html'")
    # Display the force plot in the notebook or web browser
    shap.force_plot(explainer.expected_value[1], shap_values[1], X_test_sample)


## Not working yet 
# Generate and display SHAP summary plot if shap_values is not None
if shap_values is not None:
    print(f"Number of features in X_test_sample: {X_test_sample.shape[1]}")
    print(f"Number of features in shap_values: {len(shap_values[0])}")
    #print(f"Shape of shap_values[0]: {np.array(shap_values[0]).shape}")
    #print(f"Shape of shap_values[1]: {np.array(shap_values[1]).shape}")
    shap.summary_plot(shap_values[1], X_test_sample, feature_names=feature_columns, plot_type="bar")
    plt.savefig('shap_summary_plot.png')  # Save the plot to a file
    print("SHAP summary plot saved as 'shap_summary_plot.png'")
    plt.show() """

# Function to generate SHAP summary plot for a specific activity
def generate_shap_summary_for_activity(activity_id, X, y, model):
    """
    Generate SHAP summary plot for a specific activity.

    Args:
        activity_id (int): The activity ID to analyze.
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target data.
        model: The trained model.
    """
    activity = ACTIVITY_MAPPING[activity_id]
    # Align indices of X and y
    X, y = X.align(y, join='inner', axis=0)
    
    # Filter the data for the specific activity
    activity_indices = y == activity_id
    X_activity = X[activity_indices]
    y_activity = y[activity_indices]

    # Check if there are samples for the specific activity
    if X_activity.empty:
        print(f"No samples found for activity '{activity}'. Skipping SHAP summary plot.")
        return

    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values_activity = explainer.shap_values(X_activity)

    # Check if the model is multi-class
    if isinstance(shap_values_activity, list):
        shap_values_activity = shap_values_activity[1]  # Select the SHAP values for the specific class

    # Generate SHAP summary plot
    shap.summary_plot(shap_values_activity, X_activity, feature_names=X.columns, plot_type="bar")
    plt.title(f"SHAP Summary Plot for Activity: {activity}")
    plt.savefig(f'shap_summary_plot_{activity}.png')
    print(f"SHAP summary plot for activity '{activity}' saved as 'shap_summary_plot_{activity}.png'")

# Generate SHAP summary plots for all activities
unique_activities = y_test.unique()
for activity_id in unique_activities:
    generate_shap_summary_for_activity(activity_id, X_test_sample, y_test, loaded_model)






#Once we find indentify the important sensor, we select them and train a new model on these sensors. 
#Something like this:
""" # Identify important sensors using SHAP summary plot
important_sensors = ["Sensor1", "Sensor2", "Sensor3"]  # Replace with actual important sensors

# Select necessary sensors
X_important = X[important_sensors]

# Train and evaluate the new model
model_important, X_train_important, X_test_important, y_train_important, y_test_important = train_decision_tree(X_important, y)
evaluate_model(model_important, X_test_important, y_test_important)

# Save the new model using joblib
joblib.dump(model_important, "model_important.joblib")
print("Model with important sensors saved as 'model_important.joblib'") """