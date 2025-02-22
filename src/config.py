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
