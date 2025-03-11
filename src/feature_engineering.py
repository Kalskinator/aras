import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Union, Tuple, Optional
from src.config import ACTIVITY_FREQUENT_MAPPING


def map_activities_to_categories(df):
    """
    Maps detailed activities to high-level activity categories for a resident.

    Args:
        df: DataFrame containing activity data

    Returns:
        DataFrame with mapped activity categories for the specified resident
    """
    df_new = df.copy()

    # Map activities to categories
    df_new["Activity_R1"] = df_new["Activity_R1"].map(ACTIVITY_FREQUENT_MAPPING)
    df_new["Activity_R2"] = df_new["Activity_R2"].map(ACTIVITY_FREQUENT_MAPPING)

    return df_new


def bin_data_by_time_window_with_activities(df, sensor_columns, window_size):
    """
    Bin the sensor data into time windows and include most common activity per bin.
    Groups by both day and time to get proper activation counts.

    Args:
        df: DataFrame containing sensor and activity data
        sensor_columns: List of sensor column names to bin
        window_size: Size of time window in seconds to bin data into

    Returns:
        DataFrame with binned sensor values and most common activity per resident
    """
    df_new = df.copy()

    # Create time bins
    df_new["Time"] = (df_new["Time"] // window_size) * window_size

    # Group by both day and time
    grouped = df_new.groupby(["Day", "Time"])

    # Handle sensor columns - sum values in each bin and rename with window size
    sensor_data = grouped[sensor_columns].sum().reset_index()
    sensor_data = sensor_data.rename(
        columns={col: f"{col}_{window_size}s" for col in sensor_columns}
    )

    # Get most common activity for each resident in each time bin
    r1_activities = grouped["Activity_R1"].agg(lambda x: x.mode()[0]).reset_index()
    r2_activities = grouped["Activity_R2"].agg(lambda x: x.mode()[0]).reset_index()

    # Merge sensor data with activities
    result = sensor_data.merge(r1_activities, on=["Day", "Time"])
    result = result.merge(r2_activities, on=["Day", "Time"])

    return result


def add_time_of_day_features(df):
    """
    Add time of day features based on the Time column.

    Args:
        df: DataFrame containing the time column
        time_column: Name of the column containing time information

    Returns:
        DataFrame with added time features
    """

    df["minute"] = (df["Time"] / 60).astype(int)
    df["hour"] = (df["Time"] / 3600).astype(int)

    # # Calculate part of day
    # df["part_of_day"] = pd.cut(
    #     df["hour_of_day"],
    #     bins=[0, 6, 12, 18, 24],
    #     labels=["night", "morning", "afternoon", "evening"],
    #     include_lowest=True,
    # )

    # Calculate cyclic features for minutes and hours
    # These help the model understand the cyclic nature of time
    df["minute_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["minute_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # if "Day" in df.columns:
    #     # Note: This assumes Day 1 is a specific weekday
    #     # You'd need to adjust based on your knowledge of when Day 1 actually started
    #     df["day_of_week"] = ((df["Day"] - 1) % 7) + 1  # 1-7 where 1 is Monday

    #     # Add cyclic features for day of week
    #     df["day_sin"] = np.sin(2 * np.pi * (df["day_of_week"] - 1) / 7)
    #     df["day_cos"] = np.cos(2 * np.pi * (df["day_of_week"] - 1) / 7)

    return df


def add_time_window_features(
    df,
    sensor_columns,
    window_sizes,
    aggregations,
):
    """
    Add rolling window features for sensor data.

    Args:
        df: DataFrame containing the sensor data
        sensor_columns: List of sensor column names
        window_sizes: List of window sizes in seconds
        aggregations: List of aggregation functions to apply

    Returns:
        DataFrame with added window features
    """

    for window_size in window_sizes:
        for column in sensor_columns:
            # Create a rolling window
            rolling = df[column].rolling(window=window_size, min_periods=1)

            # Apply aggregation functions
            for agg in aggregations:
                if agg == "mean":
                    df[f"{column}_mean_{window_size}s"] = rolling.mean()
                elif agg == "std":
                    df[f"{column}_std_{window_size}s"] = rolling.std().fillna(0)
                elif agg == "max":
                    df[f"{column}_max_{window_size}s"] = rolling.max()
                elif agg == "min":
                    df[f"{column}_min_{window_size}s"] = rolling.min()
                elif agg == "sum":
                    df[f"{column}_sum_{window_size}s"] = rolling.sum()
                elif agg == "count_changes":
                    # Count the number of times the sensor changes state
                    changes = (df[column].diff() != 0).astype(int)
                    df[f"{column}_changes_{window_size}s"] = changes.rolling(
                        window=window_size, min_periods=1
                    ).sum()

    return df


def engineer_features(df: pd.DataFrame, sensor_columns: List[str]) -> pd.DataFrame:
    """
    Apply all temporal feature engineering techniques to the dataframe.

    Args:
        df: DataFrame containing the sensor and activity data
        sensor_columns: List of all sensor column names

    Returns:
        DataFrame with all temporal features added
    """
    print("Performing feature engineering...")

    print("Mapping activities to categories...")
    df = map_activities_to_categories(df)

    print("Binning sensor data by time window with activities...")
    df = bin_data_by_time_window_with_activities(df, sensor_columns, 60)

    # Add time of day features
    # df = add_time_of_day_features(df)

    # print("Adding time window features...")
    # Add time window features
    # df = add_time_window_features(
    #     df,
    #     sensor_columns,
    #     # window_sizes=[30, 60, 300, 600],
    #     window_sizes=[600],
    #     aggregations=["mean", "std", "max", "min", "sum"],
    # )

    # print("Saving sample of engineered features to CSV...")
    # df.to_csv("engineered_features.csv", index=False)
    # print("Features sample saved successfully.")

    return df
