import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

from src.config import DATA_DIR_HOUSE_A, SENSOR_COLUMNS_HOUSE_A
from src.Data.data_loader import load_all_data, load_day_data
from src.models import get_model, MODEL_REGISTRY


def parse_arguments():
    parser = argparse.ArgumentParser(description="Activity prediction model")

    # Model selection argument - allows multiple models
    parser.add_argument(
        "--models",
        nargs="+",
        default=["decision_tree"],
        choices=[
            "all",
            "knn",
            "decision_tree",
            "svm",
            "logistic_regression",
            "random_forest",
            "lightgbm",
            "gradient_boosting",
            "catboost",
            "xgboost",
            "gaussiannb",
        ],
        help='One or more models to use (use "all" for all models)',
    )

    # Resident selection argument
    parser.add_argument(
        "--resident",
        type=str,
        default="R1",
        choices=["R1", "R2"],
        help="Resident to analyze (R1 or R2)",
    )

    parser.add_argument(
        "--save_models",
        action="store_true",
        help="Save trained models to disk",
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle 'all' option
    if "all" in args.models:
        args.models = [
            "decision_tree",
            "knn",
            "svm",
            "logistic_regression",
            "random_forest",
            "lightgbm",
            "gradient_boosting",
            "catboost",
            "xgboost",
            "gaussiannb",
        ]

    return args


def main(args):
    print(f"Selected model: {args.models}")
    print(f"Selected resident: {args.resident}")

    print(f"Loading data from {DATA_DIR_HOUSE_A}")
    df = load_all_data(DATA_DIR_HOUSE_A)
    # df = load_day_data(DATA_DIR_HOUSE_A / "DAY_1.txt")

    feature_columns = ["Time"] + SENSOR_COLUMNS_HOUSE_A
    X = df[feature_columns]

    target_column = f"Activity_{args.resident}"
    y = df[target_column]

    # Results storage
    results = {}
    for model_name in args.models:
        print(f"\n{'-'*40}\nTraining {model_name}...\n{'-'*40}")

        model = get_model(model_name)

        # Train model
        _, X_test, _, y_test = model.train(X, y)

        # Evaluate model
        accuracy = model.evaluate(X_test, y_test, print_report=False)

        # Store results
        results[model_name] = accuracy

        if args.save_models:
            # Create filename with resident info
            artifacts_dir = os.path.join("artifacts", "models", f"{model_name}_{args.resident}")
            os.makedirs(artifacts_dir, exist_ok=True)

            # Save with resident info in path
            model.save(artifacts_dir=artifacts_dir, accuracy=accuracy)

    # Print summary of results
    print(f"\n{'-'*40}\nResults summary\n{'-'*40}")
    for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name}: {accuracy:.4f}")


if __name__ == "__main__":
    main(parse_arguments())
