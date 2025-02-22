import pandas as pd


def convert_txt_to_csv(input_txt, output_csv):
    """
    Convert sensor data from txt to csv format

    Args:
        input_txt (str): Path to input txt file
        output_csv (str): Path to output csv file
    """
    # Create column names based on the sensor types
    sensor_cols = [
        "Ph1_Wardrobe",
        "Ph2_Convertible_Couch",
        "Ir1_TV_receiver",
        "Fo1_Couch",
        "Fo2_Couch",
        "Di3_Chair",
        "Di4_Chair",
        "Ph3_Fridge",
        "Ph4_Kitchen_Drawer",
        "Ph5_Wardrobe",
        "Ph6_Bathroom_Cabinet",
        "Co1_House_Door",
        "Co2_Bathroom_Door",
        "Co3_Shower_Cabinet_Door",
        "So1_Hall",
        "So2_Kitchen",
        "Di1_Tap",
        "Di2_Water_Closet",
        "Te1_Kitchen",
        "Fo3_Bed",
        "Resident_1_Activity",
        "Resident_2_Activity",
    ]
    # Create a dictionary mapping activity IDs to their descriptions
    activity_mapping = {
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

    # Read the space-separated txt file with no header
    df = pd.read_csv(input_txt, sep="\s+", header=None, names=sensor_cols)

    # Map activity IDs to descriptions for both residents
    # df["Resident_1_Activity"] = df["Resident_1_Activity"].map(activity_mapping)
    # df["Resident_2_Activity"] = df["Resident_2_Activity"].map(activity_mapping)

    # Save to CSV
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    # Example usage
    convert_txt_to_csv("House A/DAY_1.txt", "House A/DAY_1.csv")
