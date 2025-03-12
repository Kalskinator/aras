import os
import pandas as pd
import csv


class ResultsSaver:
    """
    Helper class for saving model evaluation results to CSV files.
    Maintains a consistent structure for comparing results across different houses and residents.
    """

    def __init__(self, output_dir):

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_results(self, model_name, metric_name, results_dict):
        """
        Save evaluation results to a CSV file.

        Args:
            model_name: Name of the model (for example: "KNN", "RandomForest")
            metric_name: Metric variant (for example: "Manhattan", "Euclidean" for KNN)
            results_dict: Dictionary with keys like "House A - R1" and values as tuples
                                of (accuracy, precision, recall, f1_score)
        """
        # Create the output directory for this model if it doesn't exist
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Define the CSV file path
        csv_path = os.path.join(model_dir, f"{metric_name}.csv")

        # Create the dataframe with the standardized structure
        data = {
            metric_name: ["Accuracy ", "Precision", "Recall", "F1-Score"],
            "House A - R1": [0, 0, 0, 0],
            "House A - R2": [0, 0, 0, 0],
            "House B - R1": [0, 0, 0, 0],
            "House B - R2": [0, 0, 0, 0],
        }

        # Fill in the results from the provided dictionary
        for key, value in results_dict.items():
            if key in data:
                accuracy, precision, recall, fscore = value
                data[key] = [
                    round(accuracy, 4),
                    round(precision, 4),
                    round(recall, 4),
                    round(fscore, 4),
                ]

        # Create and save the DataFrame
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    def update_results(self, model_name, metric_name, house, resident, results):
        """
        Update an existing CSV file with new results for a specific house and resident.
        """

        # Define the CSV file path
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        csv_path = os.path.join(model_dir, f"{metric_name}.csv")

        # Column name in the format "House A - R1"
        column_name = f"House {house} - {resident}"

        # If file exists, update it; otherwise create a new one
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            df[column_name] = [
                round(results[0], 4),  # accuracy
                round(results[1], 4),  # precision
                round(results[2], 4),  # recall
                round(results[3], 4),  # f1_score
            ]

            df.to_csv(csv_path, index=False)
            print(f"Updated results for {column_name} in {csv_path}")
        else:
            # Create a new file with this result
            results_dict = {column_name: results}
            self.save_results(model_name, metric_name, results_dict)

    def combine_results(self, model_name, output_filename="combined_results.csv"):
        """
        Combine results from multiple metric files for a model into a single CSV.
        """
        model_dir = os.path.join(self.output_dir, model_name)
        if not os.path.exists(model_dir):
            print(f"No results directory found for {model_name}")
            return

        # Get all CSV files in the model directory
        csv_files = [f for f in os.listdir(model_dir) if f.endswith(".csv")]

        if not csv_files:
            print(f"No results found for {model_name}")
            return

        # Combine the dataframes
        combined_data = []

        for csv_file in csv_files:
            file_path = os.path.join(model_dir, csv_file)
            df = pd.read_csv(file_path)

            # Extract metric name (filename without extension)
            metric_name = os.path.splitext(csv_file)[0]

            # Add metric name to the data
            df["Metric"] = metric_name
            combined_data.append(df)

        # Concatenate all dataframes
        combined_df = pd.concat(combined_data)

        # Save the combined results
        output_path = os.path.join(self.output_dir, output_filename)
        combined_df.to_csv(output_path, index=False)
        print(f"Combined results saved to {output_path}")
