import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Union, Tuple, Optional


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


def engineer_temporal_features(
    df: pd.DataFrame,
    sensor_columns: List[str],
) -> pd.DataFrame:
    """
    Apply all temporal feature engineering techniques to the dataframe.

    Args:
        df: DataFrame containing the sensor and activity data
        sensor_columns: List of all sensor column names

    Returns:
        DataFrame with all temporal features added
    """
    print("Performing feature engineering...")

    print("Adding time of day features...")
    # Add time of day features
    df = add_time_of_day_features(df)

    print("Adding time window features...")
    # Add time window features
    df = add_time_window_features(
        df,
        sensor_columns,
        window_sizes=[30, 60, 300, 600],
        aggregations=["mean", "std", "max", "min", "sum"],
    )

    print("Dropping original sensor columns...")
    df = df.drop(sensor_columns, axis=1)

    print(df.tail())

    return df
