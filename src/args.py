import argparse
from src.config import MODEL_CATEGORIES


def expand_model_categories(models):
    """Expand model category names into individual model names."""
    model_categories = MODEL_CATEGORIES
    expanded_models = []
    for model in models:
        if model in model_categories:
            expanded_models.extend(model_categories[model])
        else:
            expanded_models.append(model)

    return list(dict.fromkeys(expanded_models))


def parse_arguments():
    model_help_text = (
        "Specify one or more models to use.\n"
        'Use "all" for all models, or a category name:\n'
        + "\n".join(f"  - {category}" for category in sorted(MODEL_CATEGORIES.keys()))
        + "\n\nOr individual models:\n"
        + "\n".join(
            f"  - {model}"
            for model in sorted(
                set(model for models in MODEL_CATEGORIES.values() for model in models)
            )
        )
    )

    parser = argparse.ArgumentParser(
        description="Activity prediction model", formatter_class=argparse.RawTextHelpFormatter
    )

    # Model selection argument - allows multiple models
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        default=["decision_tree"],
        metavar="MODEL",
        help=model_help_text,
    )

    # Resident selection argument
    parser.add_argument(
        "-r",
        "--resident",
        type=str,
        default="R1",
        choices=["R1", "R2"],
        help="Resident to analyze (R1 or R2)",
    )

    parser.add_argument(
        "-ho",
        "--house",
        type=str,
        default="A",
        choices=["A", "B"],
        help="House to analyze (A or B)",
    )

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="all",
        choices=["all", "other"],
        help="Dataset to use: 'all' for complete dataset or 'other' for filtered data",
    )

    parser.add_argument(
        "-f",
        "--feature_engineering",
        action="store_true",
        help="Perform feature engineering",
    )

    parser.add_argument(
        "-t",
        "--training",
        type=str,
        choices=["default", "cross_validation", "temporal_cv"],
        default="default",
        help="Training method (default, cross_validation, temporal_cv)",
    )

    parser.add_argument(
        "-s",
        "--save_models",
        action="store_true",
        help="Save trained models to disk",
    )

    # # Add arguments for time-based split
    # parser.add_argument(
    #     "--train_days",
    #     type=int,
    #     default=7,
    #     metavar="N",
    #     help="Number of days to use for training",
    # )
    # parser.add_argument(
    #     "--val_days",
    #     type=int,
    #     default=2,
    #     metavar="N",
    #     help="Number of days to use for validation",
    # )
    # parser.add_argument(
    #     "--test_days",
    #     type=int,
    #     default=2,
    #     metavar="N",
    #     help="Number of days to use for testing",
    # )

    parser.add_argument(
        "-p",
        "--print_report",
        action="store_true",
        help="Print the classification report",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with more verbose logging",
    )

    parser.add_argument(
        "--use-smote",
        action="store_true",
        help="Use SMOTE for oversampling minority classes",
    )

    parser.add_argument(
        "--smote-strategy",
        type=str,
        choices=["auto", "minority", "not majority", "all"],
        default="auto",
        help="Strategy for SMOTE oversampling",
    )

    # Parse arguments
    args = parser.parse_args()

    # Expand model categories to individual models
    args.models = expand_model_categories(args.models)

    return args
