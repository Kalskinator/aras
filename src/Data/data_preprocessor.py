from src.Data.data_loader import DataLoader
import logging

from src.config import (
    DATA_DIR_HOUSE_A,
    DATA_DIR_HOUSE_B,
    SENSOR_COLUMNS_HOUSE_A,
    SENSOR_COLUMNS_HOUSE_B,
    ACTIVITY_COLUMNS,
)


class DataPreprocessor:
    def __init__(self):
        pass

    def prepare_data(resident, data, house):
        """
        Prepare data for a specific resident without feature engineering.

        Args:
            resident: Resident ID ('R1' or 'R2')
            data: Type of data to load ('all' or other)
            house: House ID ('A' or 'B')

        Returns:
            Tuple of (X, y) where X is features and y is target variable
        """
        data_dir = DATA_DIR_HOUSE_A if house == "A" else DATA_DIR_HOUSE_B
        sensor_columns = SENSOR_COLUMNS_HOUSE_A if house == "A" else SENSOR_COLUMNS_HOUSE_B

        logging.info(f"Loading data from {data_dir}")
        if data == "all":
            df = DataLoader.load_all_data(data_dir, sensor_columns + ACTIVITY_COLUMNS)
        else:
            df = DataLoader.load_day_data(
                data_dir / f"DAY_1.txt", sensor_columns + ACTIVITY_COLUMNS
            )

        other_resident = "R1" if resident == "R2" else "R2"
        feature_columns = ["Time"] + sensor_columns + [f"Activity_{other_resident}"]
        logging.debug(f"Feature columns: {feature_columns}")

        X = df[feature_columns]
        y = df[f"Activity_{resident}"]

        return X, y

    def prepare_data_with_engineering(resident, data, house, engineer_features):
        """
        Prepare data for a specific resident with feature engineering.

        Args:
            resident: Resident ID ('R1' or 'R2')
            data: Type of data to load ('all' or other)
            house: House ID ('A' or 'B')
            engineer_temporal_features: Function for temporal feature engineering

        Returns:
            Tuple of (X, y) where X is features and y is target variable
        """
        data_dir = DATA_DIR_HOUSE_A if house == "A" else DATA_DIR_HOUSE_B
        sensor_columns = SENSOR_COLUMNS_HOUSE_A if house == "A" else SENSOR_COLUMNS_HOUSE_B

        logging.info(f"Loading data from {data_dir}")
        if data == "all":
            df = DataLoader.load_all_data(data_dir, sensor_columns + ACTIVITY_COLUMNS)
        else:
            df = DataLoader.load_day_data(
                data_dir / f"DAY_1.txt", sensor_columns + ACTIVITY_COLUMNS
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

        # X.to_csv("final_features.csv", index=False)

        # Print the new feature set size
        logging.info(f"Original feature count: {len(feature_columns)}")
        logging.info(f"New feature count after engineering: {X.shape[1]}")
        logging.info(f"Added {X.shape[1] - len(feature_columns)} new features")
        logging.debug(f"Features: {X.columns.tolist()}")

        y = engineered_df[f"Activity_{resident}"]

        return X, y
