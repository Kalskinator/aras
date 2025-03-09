# Importing necessary packages
import xgboost as xgb
import shap
import pandas as pd
from sklearn.model_selection import train_test_split

# Loading the abalone dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
columns = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "WholeWeight",
    "ShuckedWeight",
    "VisceraWeight",
    "ShellWeight",
    "Rings",
]
abalone_data = pd.read_csv(url, header=None, names=columns)

# Data preprocessing and feature engineering
# Assuming you want to predict the number of rings, which is a continuous target variable
X = abalone_data.drop("Rings", axis=1)
y = abalone_data["Rings"]

# Convert categorical feature 'Sex' to numerical using one-hot encoding
X = pd.get_dummies(X, columns=["Sex"], drop_first=True)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating an XGBRegressor model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Save the XGBoost model in binary format
model.save_model("model.json")

# Initialize the SHAP JavaScript library
shap.initjs()

# Load the model from the saved binary file
loaded_model = xgb.XGBRegressor()
loaded_model.load_model("model.json")

# SHAP Explainer
explainer = shap.Explainer(loaded_model)
shap_values = explainer(X_test)

print(f"Shape of SHAP values: {shap_values.shape}")

# Waterfall plot for the first observation
# shap.waterfall_plot(shap_values[0])
