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
        "--models",
        nargs="+",
        default=["decision_tree"],
        metavar="MODEL",
        help=model_help_text,
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

    # Add arguments for time-based split
    parser.add_argument(
        "--train_days",
        type=int,
        default=7,
        metavar="N",
        help="Number of days to use for training",
    )
    parser.add_argument(
        "--val_days",
        type=int,
        default=2,
        metavar="N",
        help="Number of days to use for validation",
    )
    parser.add_argument(
        "--test_days",
        type=int,
        default=2,
        metavar="N",
        help="Number of days to use for testing",
    )

    parser.add_argument(
        "--print_report",
        action="store_true",
        help="Print the classification report",
    )

    # Parse arguments
    args = parser.parse_args()

    # Expand model categories to individual models
    args.models = expand_model_categories(args.models)

    # # Convert 'none' to None for feature selection
    # if args.feature_selection == "none":
    #     args.feature_selection = None

    return args
