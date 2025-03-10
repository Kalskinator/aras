from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR_HOUSE_A = ROOT_DIR / "src" / "data" / "House_A"

# Sensor and activity column definitions
SENSOR_COLUMNS_HOUSE_A = [
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

ACTIVITY_COLUMNS = ["Activity_R1", "Activity_R2"]

# Activity mapping
ACTIVITY_MAPPING = {
    1: "Other",
    2: "Going Out",
    3: "Preparing Breakfast",
    4: "Having Breakfast",
    5: "Preparing Lunch",
    6: "Having Lunch",
    7: "Preparing Dinner",
    8: "Having Dinner",
    9: "Washing Dishes",
    10: "Having Snack",
    11: "Sleeping",
    12: "Watching TV",
    13: "Studying",
    14: "Having Shower",
    15: "Toileting",
    16: "Napping",
    17: "Using Internet",
    18: "Reading Book",
    19: "Laundry",
    20: "Shaving",
    21: "Brushing Teeth",
    22: "Talking on the Phone",
    23: "Listening to Music",
    24: "Cleaning",
    25: "Having Conversation",
    26: "Having Guest",
    27: "Changing Clothes",
}

MODEL_CATEGORIES = {
    "ensemble_models": ["lightgbm", "gradient_boosting", "catboost", "xgboost"],
    "bayes_models": ["gaussiannb"],
    "linear_models": ["svm", "logistic_regression"],
    "neighbors_models": ["knn"],
    "tree_models": ["decision_tree", "random_forest"],
}
