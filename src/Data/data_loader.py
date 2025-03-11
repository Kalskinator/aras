import pandas as pd


def load_day_data(file_path, names):
    df = pd.read_csv(file_path, sep="\s+", header=None, names=names)
    df["Time"] = range(1, len(df) + 1)
    return df


def load_all_data(data_dir, names):
    all_data = []
    for day_file in sorted(data_dir.glob("DAY_*.txt"), key=lambda x: int(x.stem.split("_")[1])):
        # print(f"Loading {day_file}")
        day_data = load_day_data(day_file, names)
        day_data["Day"] = int(day_file.stem.split("_")[1])  # Add day number
        all_data.append(day_data)
    print(f"Loaded {len(all_data)} days of data")
    return pd.concat(all_data, ignore_index=True)


def load_data_with_time_split(data_dir, train_days=7, val_days=2, test_days=2):
    """
    Load data with time-based split into training, validation, and test sets.

    Args:
        data_dir: Directory containing the day files
        train_days: Number of days for training (default: 7)
        val_days: Number of days for validation (default: 2)
        test_days: Number of days for testing (default: 2)

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    df = load_all_data(data_dir)

    df = df.sort_values("Day")

    train_end_day = train_days
    val_end_day = train_end_day + val_days

    # Split the data
    train_data = df[df["Day"] <= train_end_day]
    val_data = df[(df["Day"] > train_end_day) & (df["Day"] <= val_end_day)]
    test_data = df[df["Day"] > val_end_day]

    print(f"Training data: Days 1-{train_end_day}")
    print(f"Validation data: Days {train_end_day+1}-{val_end_day}")
    print(f"Test data: Days {val_end_day+1}-{val_end_day+test_days}")

    return train_data, val_data, test_data


def prepare_data(
    resident,
    data,
    house,
    data_dir_house_a,
    data_dir_house_b,
    sensor_columns_house_a,
    sensor_columns_house_b,
    activity_columns,
):
    """
    Prepare data for a specific resident without feature engineering.

    Args:
        resident: Resident ID ('R1' or 'R2')
        data: Type of data to load ('all' or other)
        house: House ID ('A' or 'B')
        data_dir_house_a: Directory path for house A data
        data_dir_house_b: Directory path for house B data
        sensor_columns_house_a: List of sensor column names for house A
        sensor_columns_house_b: List of sensor column names for house B
        activity_columns: List of activity column names

    Returns:
        Tuple of (X, y) where X is features and y is target variable
    """
    data_dir = data_dir_house_a if house == "A" else data_dir_house_b
    sensor_columns = sensor_columns_house_a if house == "A" else sensor_columns_house_b

    print(f"Loading data from {data_dir}")
    if data == "all":
        df = load_all_data(data_dir, sensor_columns + activity_columns)
    else:
        df = load_day_data(data_dir / f"DAY_1.txt", sensor_columns + activity_columns)

    other_resident = "R1" if resident == "R2" else "R2"
    feature_columns = ["Time"] + sensor_columns + [f"Activity_{other_resident}"]
    print(f"Feature columns: {feature_columns}")

    X = df[feature_columns]
    y = df[f"Activity_{resident}"]

    return X, y


def prepare_data_with_engineering(
    resident,
    data,
    house,
    data_dir_house_a,
    data_dir_house_b,
    sensor_columns_house_a,
    sensor_columns_house_b,
    activity_columns,
    engineer_temporal_features,
):
    """
    Prepare data for a specific resident with feature engineering.

    Args:
        resident: Resident ID ('R1' or 'R2')
        data: Type of data to load ('all' or other)
        house: House ID ('A' or 'B')
        data_dir_house_a: Directory path for house A data
        data_dir_house_b: Directory path for house B data
        sensor_columns_house_a: List of sensor column names for house A
        sensor_columns_house_b: List of sensor column names for house B
        activity_columns: List of activity column names
        engineer_temporal_features: Function for temporal feature engineering

    Returns:
        Tuple of (X, y) where X is features and y is target variable
    """
    data_dir = data_dir_house_a if house == "A" else data_dir_house_b
    sensor_columns = sensor_columns_house_a if house == "A" else sensor_columns_house_b

    print(f"Loading data from {data_dir}")
    if data == "all":
        df = load_all_data(data_dir, sensor_columns + activity_columns)
    else:
        df = load_day_data(data_dir / f"DAY_1.txt", sensor_columns + activity_columns)

    other_resident = "R1" if resident == "R2" else "R2"
    feature_columns = ["Time"] + sensor_columns + [f"Activity_{other_resident}"]

    engineered_df = engineer_temporal_features(
        df=df,
        sensor_columns=sensor_columns,
        binary_sensor_columns=sensor_columns,
        activity_column=f"Activity_{other_resident}",  # Use other resident's activity
    )

    # Remove the activity columns from features
    X = engineered_df.drop(
        [f"Activity_{other_resident}", f"Activity_{resident}"], axis=1, errors="ignore"
    )

    # Print the new feature set size
    print(f"Original feature count: {len(feature_columns)}")
    print(f"New feature count after engineering: {X.shape[1]}")
    print(f"Added {X.shape[1] - len(feature_columns)} new features")

    y = engineered_df[f"Activity_{resident}"]

    return X, y
