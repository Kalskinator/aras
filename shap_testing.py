from src.Data.data_loader import load_all_data, load_day_data
from src.config import DATA_DIR_HOUSE_A, SENSOR_COLUMNS_HOUSE_A, ACTIVITY_MAPPING
from src.models.decision_tree import train_decision_tree, evaluate_model
import shap
import numpy as np
import pandas as pd

df = load_all_data(DATA_DIR_HOUSE_A)
# df = load_day_data(DATA_DIR_HOUSE_A / "DAY_1.txt")
# df2 = load_day_data(DATA_DIR_HOUSE_A / "DAY_2.txt")


feature_columns = ["Time"] + SENSOR_COLUMNS_HOUSE_A

X = df[feature_columns]  # Use only sensor columns
y = df["Activity_R2"]  # Predict activities

model, X_train, X_test, y_train, y_test = train_decision_tree(X, y)
evaluate_model(model, X_test, y_test)

activities = list(map(lambda x: ACTIVITY_MAPPING[x], model.classes_))
print(activities)


# Initialize the SHAP JavaScript library
shap.initjs()

# SHAP Explainer
print("Initializing SHAP explainer")
explainer = shap.TreeExplainer(model)

print("Calculating SHAP values")
try:

    shap_values = explainer(X_test)
    print("test")
    # print("SHAP values calculated")
    # print(f"Shape of SHAP values: {shap_values}")
    # print(f"Shape of X_test: {X_test.shape}")
    # print(shap_values[0, 0].base_values)
    shap.plots.waterfall(shap_values[0, :, 0])
    # shap.waterfall_plot(shap_values[0])

    # expl = shap_values[0, 0]
    # expl.base_values = expl.base_values[0]  # extract the first base value as a scalar
    # shap.plots.waterfall(expl)

    # expl = shap_values[2, 0]
    # # Replace the base_values array with its first element
    # expl.base_values = expl.base_values[0]
    # expl.data = X_test[2]
    # shap.plots.waterfall(expl)

    print("Finished")
except TimeoutError:
    print("SHAP calculation timed out")
    shap_values = None
