import sys
import os
import numpy as np
<<<<<<< Updated upstream
=======
from tqdm import tqdm
from utils.progress_bar_helper import ProgressBarHelper
>>>>>>> Stashed changes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    DATA_DIR_HOUSE_A,
    DATA_DIR_HOUSE_B,
    SENSOR_COLUMNS_HOUSE_A,
    SENSOR_COLUMNS_HOUSE_B,
    ACTIVITY_COLUMNS,
)
from src.Data.data_loader import (
    load_data_with_time_split,
    load_day_data,
    load_all_data,
    prepare_data,
    prepare_data_with_engineering,
)
from src.models import get_model
from src.args import parse_arguments
from src.feature_engineering import engineer_temporal_features


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
    print(f"Selected house: {args.house}")

    # Prepare data
    if not args.feature_engineering:
        X, y = prepare_data(
            args.resident,
            args.data,
            args.house,
            DATA_DIR_HOUSE_A,
            DATA_DIR_HOUSE_B,
            SENSOR_COLUMNS_HOUSE_A,
            SENSOR_COLUMNS_HOUSE_B,
            ACTIVITY_COLUMNS,
        )
    if args.feature_engineering:
        X, y = prepare_data_with_engineering(
            args.resident,
            args.data,
            args.house,
            DATA_DIR_HOUSE_A,
            DATA_DIR_HOUSE_B,
            SENSOR_COLUMNS_HOUSE_A,
            SENSOR_COLUMNS_HOUSE_B,
            ACTIVITY_COLUMNS,
            engineer_temporal_features,
        )

# Define hyperparameter grid for LightGBM
    """     param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100],
        'max_depth': [-1, 10, 20],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    } """

    results = {}
    for model_name in args.models:
        model = get_model(model_name)
        # Commenting out the hyperparameter tuning call
        # if model_name == "lightgbm":
        #     best_params, best_score = model.tune_hyperparameters(X, y, param_grid, n_iter=25)
        #     print(f"Best parameters for {model_name}: {best_params}")
        #     print(f"Best cross-validation score for {model_name}: {best_score:.4f}")

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
