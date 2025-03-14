import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from tqdm import tqdm

from src.utils.logging_utils import (
    setup_logger,
    log_section,
    log_results,
    log_parameters,
    disable_third_party_loggers,
    enable_debug_mode,
)

from src.models import get_model
from src.args import parse_arguments
from src.Data.data_preprocessor import DataPreprocessor
from src.feature_engineering import FeatureEngineering
from src.utils.results_saver import ResultsSaver
from src.Data.data_resampler import DataResampler
from sklearn.model_selection import TimeSeriesSplit


def train_and_evaluate_model(model_name, X, y, print_report=False, house=None, resident=None):
    """Train and evaluate a single model."""
    log_section(f"Training {model_name}...")

    model = get_model(model_name)

    X_train, X_test, y_train, y_test = model.train(X, y)

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

        metrics, model = train_and_evaluate_model(model_name, X, y, print_report, house, resident)
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


def cross_validate_with_smote(
    model_names,
    resident,
    house,
    data,
    feature_engineering=False,
    print_report=False,
    save_models=False,
    use_smote=False,
    sampling_strategy="auto",
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
        use_smote: Whether to use SMOTE for oversampling minority classes
        sampling_strategy: Strategy for SMOTE oversampling

    Returns:
        Dictionary of model results
    """
    log_section("Performing Resident-Based Cross-Validation")
    results = {}

    # Get the other resident ID for training
    other_resident = "R1" if resident == "R2" else "R2"
    logging.info(f"Training on {resident}'s data, validating on {other_resident}'s data")

    logging.info(f"Loading training data from {resident}...")
    if not feature_engineering:
        X_train, y_train = DataPreprocessor.prepare_data(resident, "all", house)
    else:
        X_train, y_train = DataPreprocessor.prepare_data_with_engineering(
            resident,
            data,
            house,
            FeatureEngineering.engineer_features,
        )

    # Load data for the validation resident (the target resident)
    logging.info(f"Loading validation data from {other_resident}...")
    if not feature_engineering:
        X_val, y_val = DataPreprocessor.prepare_data(other_resident, "all", house)
    else:
        X_val, y_val = DataPreprocessor.prepare_data_with_engineering(
            other_resident,
            data,
            house,
            FeatureEngineering.engineer_features,
        )

    # Apply SMOTE to training data if requested
    if use_smote:
        X_train, y_train = DataResampler(X_train, y_train).apply_smote(
            sampling_strategy=sampling_strategy,
            random_state=42,
            k_neighbors=5,
        )

    # For each model, train on one resident and validate on the other
    for model_name in model_names:
        log_section(f"Training {model_name} on {resident}'s data...")
        model = get_model(model_name)

        # Fit the model on the training resident's data
        if hasattr(model, "model") and model.model is None:
            # If the model needs to be initialized
            # We need to call train but ignore the train/test split it creates
            _, _, _, _ = model.train(X_train, y_train)
        else:
            # If the model is already initialized, just fit it
            model.model.fit(X_train, y_train)

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
            # Add SMOTE to the model name if it was used
            smote_suffix = "_smote" if use_smote else ""
            model_suffix = f"{model_name}_{resident}_to_{other_resident}{smote_suffix}"
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


def temporal_cross_validation_with_smote(
    model_names,
    resident,
    house,
    data,
    feature_engineering=False,
    print_report=False,
    save_models=False,
    n_splits=5,
    use_smote=True,
    sampling_strategy="auto",  # 'auto', 'minority', 'not majority', 'all', or a dictionary
):
    """
    Perform temporal cross-validation with time-based splits and SMOTE to address class imbalance.

    Args:
        model_names: List of model names to train
        resident: The target resident (R1 or R2) for evaluation
        house: The house (A or B) for data
        data: The data to use for training and validation
        feature_engineering: Whether to use feature engineering
        print_report: Whether to print classification reports
        save_models: Whether to save trained models to disk
        n_splits: Number of time-based splits to create
        use_smote: Whether to use SMOTE for oversampling minority classes
        sampling_strategy: Strategy for SMOTE oversampling

    Returns:
        Dictionary of model results
    """

    log_section("Performing Temporal Cross-Validation with SMOTE")
    results = {}

    logging.info(f"Loading data for resident {resident} from house {house}...")
    if not feature_engineering:
        X, y = DataPreprocessor.prepare_data(resident, data, house)
    else:
        X, y = DataPreprocessor.prepare_data_with_engineering(
            resident, data, house, FeatureEngineering.engineer_features
        )

    # Create time-based folds
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # For each model, perform cross-validation
    for model_name in model_names:
        log_section(f"Training {model_name} with temporal cross-validation and SMOTE")
        model_metrics = {"accuracy": [], "precision": [], "recall": [], "fscore": []}

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logging.info(f"Fold {fold+1}/{n_splits}")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            logging.info(f"Original training set size: {len(X_train)} samples")

            # Apply SMOTE to the training data only
            if use_smote:
                X_train, y_train = DataResampler(X_train, y_train).apply_smote(
                    sampling_strategy=sampling_strategy,
                    random_state=42,
                    k_neighbors=5,
                )

            logging.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

            # Train model
            model = get_model(model_name)
            if hasattr(model, "model") and model.model is None:
                # If the model needs to be initialized
                # We need to call train but ignore the train/test split it creates
                _, _, _, _ = model.train(X_train, y_train)
            else:
                # If the model is already initialized, just fit it
                model.model.fit(X_train, y_train)

            # Evaluate (on original test data, never on synthetic data)
            accuracy, precision, recall, fscore = model.evaluate(
                X_test, y_test, print_report=print_report
            )

            # Store results
            model_metrics["accuracy"].append(accuracy)
            model_metrics["precision"].append(np.mean(precision))
            model_metrics["recall"].append(np.mean(recall))
            model_metrics["fscore"].append(np.mean(fscore))

            logging.info(
                f"Fold {fold+1} Results - Accuracy: {accuracy:.4f}, F1: {np.mean(fscore):.4f}"
            )

        # Average results across splits
        final_metrics = {
            "accuracy": np.mean(model_metrics["accuracy"]),
            "precision": np.mean(model_metrics["precision"]),
            "recall": np.mean(model_metrics["recall"]),
            "fscore": np.mean(model_metrics["fscore"]),
        }

        # Log results
        logging.info(f"Average Results for {model_name}:")
        logging.info(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
        logging.info(f"  Precision: {final_metrics['precision']:.4f}")
        logging.info(f"  Recall:    {final_metrics['recall']:.4f}")
        logging.info(f"  F1 Score:  {final_metrics['fscore']:.4f}")

        results[model_name] = final_metrics

        # Save model if requested
        if save_models:
            model_suffix = f"{model_name}_{resident}_temporal_cv_smote"
            model_dir = os.path.join("src", "artifacts", "models", model_suffix)
            os.makedirs(model_dir, exist_ok=True)

            model.save(
                artifacts_dir=model_dir,
                accuracy=final_metrics["accuracy"],
                precision=final_metrics["precision"],
                recall=final_metrics["recall"],
                fscore=final_metrics["fscore"],
            )

    return results


# def LSTM_training(
#     models,
#     resident,
#     house,
#     data,
#     feature_engineering=False,
#     print_report=False,
#     save_models=False,
# ):
#     log_section("Training LSTM model")
#     from src.models.recurrent_neural_network import LSTMModel

#     logging.info(f"Loading data for resident {resident} from house {house}...")
#     if not feature_engineering:
#         X, y = DataPreprocessor.prepare_data(resident, data, house)
#     else:
#         X, y = DataPreprocessor.prepare_data_with_engineering(
#             resident, data, house, FeatureEngineering.engineer_features
#         )

#     input_shape = X.shape
#     num_classes = len(np.unique(y))

#     # Create the LSTMModel instance rather than the raw LSTMNetwork
#     model = LSTMModel(input_shape=input_shape, num_classes=num_classes)

#     # Train model
#     X_train, X_test, y_train, y_test = model.train(
#         X, y, test_size=0.3, random_state=42, epochs=50, batch_size=32
#     )

#     # Evaluate model
#     accuracy, precision, recall, fscore = model.evaluate(X_test, y_test, print_report=print_report)

#     metrics = {
#         "accuracy": accuracy,
#         "precision": np.mean(precision),
#         "recall": np.mean(recall),
#         "fscore": np.mean(fscore),
#     }

#     if save_models:
#         save_model_artifacts(
#             model,
#             "lstm",
#             resident,
#             metrics,
#             os.path.join("src", "artifacts", "models"),
#         )

#     return metrics


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
    log_section("Results Summary")
    for model_name, metrics in results.items():
        log_results(model_name, metrics)


def main(args):
    # Set up logging
    setup_logger(
        log_level=logging.INFO,
        console_output=True,
    )

    # Enable debug mode if requested
    if args.debug:
        enable_debug_mode()

    # Disable verbose logging from third party libraries
    disable_third_party_loggers()

    # Log experiment parameters
    parameters = {
        "models": args.models,
        "resident": args.resident,
        "house": args.house,
        "data": args.data,
        "feature_engineering": args.feature_engineering,
        "training_method": args.training,
        "save_models": args.save_models,
        "debug": args.debug,
    }
    log_parameters(parameters)

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
        results = cross_validate_with_smote(
            args.models,
            args.resident,
            args.house,
            args.data,
            args.feature_engineering,
            args.print_report,
            args.save_models,
            use_smote=True,
            sampling_strategy=args.smote_strategy,
        )
    elif args.training == "temporal_cv":
        # Perform temporal cross-validation
        if args.use_smote:
            results = temporal_cross_validation_with_smote(
                args.models,
                args.resident,
                args.house,
                args.data,
                args.feature_engineering,
                args.print_report,
                args.save_models,
                use_smote=True,
                sampling_strategy=args.smote_strategy,
            )
    # elif args.training == "lstm":
    #     results = LSTM_training(
    #         args.models,
    #         args.resident,
    #         args.house,
    #         args.data,
    #         args.feature_engineering,
    #         args.print_report,
    #         args.save_models,
    #     )
    print_results(results)


if __name__ == "__main__":
    main(parse_arguments())
