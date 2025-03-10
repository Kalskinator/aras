import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DATA_DIR_HOUSE_A, SENSOR_COLUMNS_HOUSE_A, ACTIVITY_COLUMNS
from src.Data.data_loader import load_data_with_time_split, load_day_data, load_all_data
from src.models import get_model
from src.args import parse_arguments


def prepare_data(resident):
    """Load and prepare data for training."""
    print(f"Loading data from {DATA_DIR_HOUSE_A}")
    df = load_all_data(DATA_DIR_HOUSE_A, SENSOR_COLUMNS_HOUSE_A + ACTIVITY_COLUMNS)
    # df = load_day_data(DATA_DIR_HOUSE_A / "DAY_1.txt", SENSOR_COLUMNS_HOUSE_A + ACTIVITY_COLUMNS)

    other_resident = "R1" if resident == "R2" else "R2"
    feature_columns = ["Time"] + SENSOR_COLUMNS_HOUSE_A + [f"Activity_{other_resident}"]
    print(f"Feature columns: {feature_columns}")

    X = df[feature_columns]
    # Prepare features and targets for each split
    # X_train = train_data[feature_columns]
    # X_val = val_data[feature_columns]
    # X_test = test_data[feature_columns]

    y = df[f"Activity_{resident}"]
    # y_train = train_data[target_column]
    # y_val = val_data[target_column]
    # y_test = test_data[target_column]

    return X, y


def train_and_evaluate_model(model_name, X, y, print_report=False):
    """Train and evaluate a single model."""
    print(f"\n{'-'*40}\nTraining {model_name}...\n{'-'*40}")

    model = get_model(model_name)
    X_train, X_test, y_train, y_test = model.train(X, y)

    accuracy, precision, recall, fscore = model.evaluate(X_test, y_test, print_report=print_report)

    return {
        "accuracy": accuracy,
        "precision": np.mean(precision),
        "recall": np.mean(recall),
        "fscore": np.mean(fscore),
    }, model


def save_model_artifacts(model, model_name, resident, metrics, artifacts_dir):
    """Save model and its metrics to disk."""
    model_dir = os.path.join(artifacts_dir, f"{model_name}_{resident}")
    os.makedirs(model_dir, exist_ok=True)

    model.save(
        artifacts_dir=model_dir,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        fscore=metrics["fscore"],
    )


def print_results(results):
    """Print summary of all model results."""
    print(f"\n{'-'*40}\nResults summary\n{'-'*40}")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['fscore']:.4f}")


def main(args):
    print(f"Selected model: {args.models}")
    print(f"Selected resident: {args.resident}")

    # Prepare data
    X, y = prepare_data(args.resident)

    # Train and evaluate models
    results = {}
    for model_name in args.models:
        metrics, model = train_and_evaluate_model(model_name, X, y, args.print_report)
        results[model_name] = metrics

        if args.save_models:
            save_model_artifacts(
                model,
                model_name,
                args.resident,
                metrics,
                os.path.join("src", "artifacts", "models"),
            )

    print_results(results)


if __name__ == "__main__":
    main(parse_arguments())
