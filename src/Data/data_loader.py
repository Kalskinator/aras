import pandas as pd
from src.config import SENSOR_COLUMNS_HOUSE_A, ACTIVITY_COLUMNS


def load_day_data(file_path):
    df = pd.read_csv(
        file_path, sep="\s+", header=None, names=SENSOR_COLUMNS_HOUSE_A + ACTIVITY_COLUMNS
    )
    df["Time"] = range(1, len(df) + 1)
    return df


def load_all_data(data_dir):
    all_data = []
    for day_file in sorted(data_dir.glob("DAY_*.txt"), key=lambda x: int(x.stem.split("_")[1])):
        # print(f"Loading {day_file}")
        day_data = load_day_data(day_file)
        # day_data["Day"] = day_file.stem.split("_")[1]
        all_data.append(day_data)
    print(f"Loaded {len(all_data)} days of data")
    return pd.concat(all_data, ignore_index=True)
