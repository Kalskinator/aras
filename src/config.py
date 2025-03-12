from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR_HOUSE_A = ROOT_DIR / "src" / "data" / "House_A"
DATA_DIR_HOUSE_B = ROOT_DIR / "src" / "data" / "House_B"

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

SENSOR_COLUMNS_HOUSE_B = [
    "co1_KitchenCupboard",
    "co2_KitchenCupboard",
    "co3_HouseDoor",
    "co4_WardrobeDoor",
    "co5_WardrobeDoor",
    "co6_ShowerCabinetDoor",
    "di2_Tap",
    "fo1_Chair",
    "fo2_Chair",
    "fo3_Chair",
    "ph1_Fridge",
    "ph2_KitchenDrawer",
    "pr1_Couch",
    "pr2_Couch",
    "pr3_Bed",
    "pr4_Bed",
    "pr5_Armchair",
    "so1_BathroomDoor",
    "so2_Kitchen",
    "so3_Closet",
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
"""
Map frequent activities to their corresponding categories.

1. Other = Other, Laundry, Cleaning
2. Sleeping = Sleeping, Napping
3. Eating = Preparing Breakfast, Having Breakfast, Preparing Lunch, Having Lunch, Preparing Dinner, Having Dinner, Having Snack
4. Personal Hygiene = Having Shower, Toileting, Brushing Teeth, Shaving, Changing Clothes
5. Going Out = Going Out
6. Relaxing = Watching TV, Studying, Using Internet, Reading Book, Talking on the Phone, Talking on Phone, Listening to Music, Having Conversation, Having Guest


"""
ACTIVITY_FREQUENT_MAPPING = {
    1: 1,  # Other
    2: 5,  # Going Out
    3: 3,  # Preparing Breakfast -> Eating
    4: 3,  # Having Breakfast -> Eating
    5: 3,  # Preparing Lunch -> Eating
    6: 3,  # Having Lunch -> Eating
    7: 3,  # Preparing Dinner -> Eating
    8: 3,  # Having Dinner -> Eating
    9: 1,  # Washing Dishes -> Other
    10: 3,  # Having Snack -> Eating
    11: 2,  # Sleeping
    12: 6,  # Watching TV -> Relaxing
    13: 6,  # Studying -> Relaxing
    14: 4,  # Having Shower -> Personal Hygiene
    15: 4,  # Toileting -> Personal Hygiene
    16: 2,  # Napping -> Sleeping
    17: 6,  # Using Internet -> Relaxing
    18: 6,  # Reading Book -> Relaxing
    19: 1,  # Laundry -> Other
    20: 4,  # Shaving -> Personal Hygiene
    21: 4,  # Brushing Teeth -> Personal Hygiene
    22: 6,  # Talking on Phone -> Relaxing
    23: 6,  # Listening to Music -> Relaxing
    24: 1,  # Cleaning -> Other
    25: 6,  # Having Conversation -> Relaxing
    26: 6,  # Having Guest -> Relaxing
    27: 4,  # Changing Clothes -> Personal Hygiene
}

MODEL_CATEGORIES = {
    "all": [
        "knn",
        "svm",
        "decision_tree",
        "logistic_regression",
        "random_forest",
        "lightgbm",
        "gradient_boosting",
        "catboost",
        "xgboost",
        "gaussiannb",
    ],
    "ensemble_models": ["lightgbm", "gradient_boosting", "catboost", "xgboost"],
    "bayes_models": ["gaussiannb"],
    "linear_models": ["svm", "logistic_regression"],
    "neighbors_models": ["knn"],
    "tree_models": ["decision_tree", "random_forest"],
}
