from src.Data.data_loader import DataLoader


class DataPreprocessor:
    def __init__(self):
        pass

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
            df = DataLoader.load_all_data(data_dir, sensor_columns + activity_columns)
        else:
            df = DataLoader.load_day_data(
                data_dir / f"DAY_1.txt", sensor_columns + activity_columns
            )

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
        engineer_features,
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
            df = DataLoader.load_all_data(data_dir, sensor_columns + activity_columns)
        else:
            df = DataLoader.load_day_data(
                data_dir / f"DAY_1.txt", sensor_columns + activity_columns
            )

        other_resident = "R1" if resident == "R2" else "R2"
        feature_columns = ["Time"] + sensor_columns + [f"Activity_{other_resident}"]

        engineered_df = engineer_features(
            df=df,
            sensor_columns=sensor_columns,
        )

        X = engineered_df.drop(
            ["Day", f"Activity_{other_resident}", f"Activity_{resident}"],
            axis=1,
        )

        X.to_csv("final_features.csv", index=False)

        # Print the new feature set size
        print(f"Original feature count: {len(feature_columns)}")
        print(f"New feature count after engineering: {X.shape[1]}")
        print(f"Added {X.shape[1] - len(feature_columns)} new features")
        print(f"Features: {X.columns.tolist()}")

        y = engineered_df[f"Activity_{resident}"]

        return X, y
