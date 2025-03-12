import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from tqdm import tqdm
from src.utils.progress_bar_helper import ProgressBarHelper

from src.models import get_model
from src.args import parse_arguments
from src.Data.data_preprocessor import DataPreprocessor
from src.feature_engineering import FeatureEngineering
from src.models.ensemble_models import LIGHTGBM_PARAM_GRID  # Import the parameter grid


def train_and_evaluate_model(model_name, X, y, print_report=False):
    """Train and evaluate a single model."""
    print(f"\n{'-'*40}\nTraining {model_name}...\n{'-'*40}")

    model = get_model(model_name)

    # Create a progress bar for the training process
    progress_bar = ProgressBarHelper(total=4, desc=f"Training {model_name}")
    X_train, X_test, y_train, y_test = model.train(X, y, progress_bar=progress_bar)
    progress_bar.close()

    accuracy, precision, recall, fscore = model.evaluate(X_test, y_test, print_report=print_report)

    return {
        "accuracy": accuracy,
        "precision": np.mean(precision),
        "recall": np.mean(recall),
        "fscore": np.mean(fscore),
    }, model


def basic_training(
    model_names,
    resident,
    house,
    data,
    feature_engineering=False,
    print_report=False,
    save_models=False,
):
    results = {}
    if not feature_engineering:
        X, y = DataPreprocessor.prepare_data(resident, data, house)
    else:
        X, y = DataPreprocessor.prepare_data_with_engineering(
            resident,
            data,
            house,
            FeatureEngineering.engineer_features,
        )

    for model_name in model_names:
        model = get_model(model_name)
        # Use the imported parameter grid
        if model_name == "lightgbm":
            best_params, best_score = model.tune_hyperparameters(
                X, y, LIGHTGBM_PARAM_GRID, n_iter=25
            )
            print(f"Best parameters for {model_name}: {best_params}")
            print(f"Best cross-validation score for {model_name}: {best_score:.4f}")

        metrics, model = train_and_evaluate_model(model_name, X, y, print_report)
        results[model_name] = metrics

        if save_models:
            save_model_artifacts(
                model,
                model_name,
                resident,
                metrics,
                os.path.join("src", "artifacts", "models"),
            )

    return results


def cross_validate_models(
    model_names,
    resident,
    house,
    data,
    feature_engineering=False,
    print_report=False,
    save_models=False,
):
    """
    Perform Group K-Fold Cross-Validation with resident-level grouping.
    Train on one resident's data and validate on the other resident's data.

    Args:
        model_names: List of model names to train
        resident: The target resident (R1 or R2) for evaluation
        house: The house (A or B) for data
        data: The data to use for training and validation
        feature_engineering: Whether to use feature engineering
        print_report: Whether to print classification reports
        save_models: Whether to save trained models to disk

    Returns:
        Dictionary of model results
    """
    print(f"\n{'-'*40}\nPerforming Resident-Based Cross-Validation\n{'-'*40}")
    results = {}

    # Get the other resident ID for training
    other_resident = "R1" if resident == "R2" else "R2"
    print(f"Training on {other_resident}'s data, validating on {resident}'s data")

    # Load data for the training resident (the other resident)
    print(f"Loading training data from {other_resident}...")
    if not feature_engineering:
        X_train, y_train = DataPreprocessor.prepare_data(other_resident, "all", house)
    else:
        X_train, y_train = DataPreprocessor.prepare_data_with_engineering(
            other_resident,
            data,
            house,
            FeatureEngineering.engineer_features,
        )

    # Load data for the validation resident (the target resident)
    print(f"Loading validation data from {resident}...")
    if not feature_engineering:
        X_val, y_val = DataPreprocessor.prepare_data(resident, "all", house)
    else:
        X_val, y_val = DataPreprocessor.prepare_data_with_engineering(
            resident,
            data,
            house,
            FeatureEngineering.engineer_features,
        )

    # For each model, train on one resident and validate on the other
    for model_name in model_names:
        print(f"\n{'-'*40}\nTraining {model_name} on {other_resident}'s data...\n{'-'*40}")
        model = get_model(model_name)

        # Fit the model on the training resident's data
        # progress_bar = ProgressBarHelper(total=4, desc=f"Training {model_name}")
        if hasattr(model, "model") and model.model is None:
            # If the model needs to be initialized
            # We need to call train but ignore the train/test split it creates
            _, _, _, _ = model.train(X_train, y_train)
            #  progress_bar=progress_bar
            # )
        else:
            # If the model is already initialized, just fit it
            model.model.fit(X_train, y_train)
            # for _ in range(4):  # Just to show progress
            #     progress_bar.update(1)
            # progress_bar.close()

        # Evaluate on the validation resident's data
        accuracy, precision, recall, fscore = model.evaluate(
            X_val, y_val, print_report=print_report
        )

        metrics = {
            "accuracy": accuracy,
            "precision": np.mean(precision),
            "recall": np.mean(recall),
            "fscore": np.mean(fscore),
        }

        results[model_name] = metrics

        # Save model if requested
        if save_models:
            model_suffix = f"{model_name}_{other_resident}_to_{resident}"
            model_dir = os.path.join("src", "artifacts", "models", model_suffix)
            os.makedirs(model_dir, exist_ok=True)

            model.save(
                artifacts_dir=model_dir,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                fscore=metrics["fscore"],
            )

    return results


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
    if not args.no_training:
        results = {}
        if args.training == "default":
            results = basic_training(
                args.models,
                args.resident,
                args.house,
                args.data,
                args.feature_engineering,
                args.print_report,
                args.save_models,
            )
        elif args.training == "cross_validation":
            # Perform resident-level cross-validation
            results = cross_validate_models(
                args.models,
                args.resident,
                args.house,
                args.data,
                args.feature_engineering,
                args.print_report,
                args.save_models,
            )
        print_results(results)


if __name__ == "__main__":
    main(parse_arguments())
