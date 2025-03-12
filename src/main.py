import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from tqdm import tqdm
from src.utils.progress_bar_helper import ProgressBarHelper
from src.utils.results_saver import ResultsSaver

from src.config import (
    DATA_DIR_HOUSE_A,
    DATA_DIR_HOUSE_B,
    SENSOR_COLUMNS_HOUSE_A,
    SENSOR_COLUMNS_HOUSE_B,
    ACTIVITY_COLUMNS,
)
from src.models import get_model
from src.args import parse_arguments
from src.Data.data_preprocessor import DataPreprocessor
from src.feature_engineering import FeatureEngineering
from src.models.ensemble_models import LIGHTGBM_PARAM_GRID  # Import the parameter grid


def train_and_evaluate_model(model_name, X, y, print_report=False, house=None, resident=None):
    """Train and evaluate a single model."""
    print(f"\n{'-'*40}\nTraining {model_name}...\n{'-'*40}")

    model = get_model(model_name)

    # Create a progress bar for the training process
    progress_bar = ProgressBarHelper(total=4, desc=f"Training {model_name}")
    X_train, X_test, y_train, y_test = model.train(X, y, progress_bar=progress_bar)
    progress_bar.close()

    accuracy, precision, recall, fscore = model.evaluate(X_test, y_test, print_report=print_report)

    # Save results if house and resident are provided
    if house and resident:

        # Get the appropriate metric name for the model
        metric_name = model_name
        if model_name == "knn":
            metric_name = model.metric.capitalize()

        # Save the results
        results_saver = ResultsSaver("Model_Results")
        results_saver.update_results(
            model_name=model_name.upper(),
            metric_name=metric_name,
            house=house,
            resident=resident,
            results=(accuracy, np.mean(precision), np.mean(recall), np.mean(fscore)),
        )

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
        X, y = DataPreprocessor.prepare_data(
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
        X, y = DataPreprocessor.prepare_data_with_engineering(
            args.resident,
            args.data,
            args.house,
            DATA_DIR_HOUSE_A,
            DATA_DIR_HOUSE_B,
            SENSOR_COLUMNS_HOUSE_A,
            SENSOR_COLUMNS_HOUSE_B,
            ACTIVITY_COLUMNS,
            FeatureEngineering.engineer_features,
        )

    if not args.no_training:
        results = {}
        if args.training == "default":
            for model_name in args.models:
                model = get_model(model_name)
                # Use the imported parameter grid
                if model_name == "lightgbm":
                    best_params, best_score = model.tune_hyperparameters(
                        X, y, LIGHTGBM_PARAM_GRID, n_iter=25
                    )
                    print(f"Best parameters for {model_name}: {best_params}")
                    print(f"Best cross-validation score for {model_name}: {best_score:.4f}")

                metrics, model = train_and_evaluate_model(
                    model_name, X, y, args.print_report, house=args.house, resident=args.resident
                )
                results[model_name] = metrics

                if args.save_models:
                    save_model_artifacts(
                        model,
                        model_name,
                        args.resident,
                        metrics,
                        os.path.join("src", "artifacts", "models"),
                    )

        elif args.training == "cross_validation":
            # results = cross_validate_models(args.models, X, y, args.print_report)
            pass
        print_results(results)


if __name__ == "__main__":
    main(parse_arguments())
