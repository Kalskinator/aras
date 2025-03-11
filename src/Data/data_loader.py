import pandas as pd


class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_day_data(file_path, names):
        df = pd.read_csv(file_path, sep="\s+", header=None, names=names)
        df["Time"] = range(1, len(df) + 1)
        return df

    @staticmethod
    def load_all_data(data_dir, names):
        all_data = []
        for day_file in sorted(data_dir.glob("DAY_*.txt"), key=lambda x: int(x.stem.split("_")[1])):
            # print(f"Loading {day_file}")
            day_data = DataLoader.load_day_data(day_file, names)
            day_data["Day"] = int(day_file.stem.split("_")[1])  # Add day number
            all_data.append(day_data)
        print(f"Loaded {len(all_data)} days of data")
        return pd.concat(all_data, ignore_index=True)

    @staticmethod
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
        df = DataLoader.load_all_data(data_dir)

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
