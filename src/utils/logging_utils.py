import logging
import os
import sys
from datetime import datetime


def setup_logger(log_level=logging.INFO, log_file=None, console_output=True):
    """
    Sets up the logger with the specified configuration.

    Args:
        log_level: Logging level (default: INFO)
        log_file: Path to the log file (default: None, creates a timestamped file in logs directory)
        console_output: Whether to output logs to console (default: True)

    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("src", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"log_{timestamp}.txt")

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Log file: {os.path.abspath(log_file)}")
    return logger


def enable_debug_mode():
    """
    Enables debug mode logging (more verbose output).
    Should be called after setup_logger.
    """
    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug("Debug logging enabled")


def disable_third_party_loggers():
    """
    Disable verbose logging from third party libraries.
    """
    # Set known verbose loggers to a higher log level
    for logger_name in ["matplotlib", "PIL", "sklearn", "tensorflow", "pandas", "numpy"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def log_section(title, level=logging.INFO):
    """
    Logs a section title with visual separation for better readability.

    Args:
        title: The section title
        level: Logging level for the section
    """
    logger = logging.getLogger()
    separator = "-" * 40
    logger.log(level, f"\n{separator}\n{title}\n{separator}")


def log_results(model_name, metrics, level=logging.INFO):
    """
    Logs model evaluation results in a formatted way.

    Args:
        model_name: Name of the model
        metrics: Dictionary containing metrics (accuracy, precision, recall, fscore)
        level: Logging level for the results
    """
    logger = logging.getLogger()
    logger.log(level, f"\n{model_name}:")
    logger.log(level, f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.log(level, f"  Precision: {metrics['precision']:.4f}")
    logger.log(level, f"  Recall:    {metrics['recall']:.4f}")
    logger.log(level, f"  F1-Score:  {metrics['fscore']:.4f}")


def log_dict(title, data_dict, level=logging.INFO):
    """
    Logs a dictionary with nice formatting.

    Args:
        title: Title for the dictionary
        data_dict: Dictionary to log
        level: Logging level
    """
    logger = logging.getLogger()
    logger.log(level, f"\n{title}:")
    for key, value in data_dict.items():
        logger.log(level, f"  {key}: {value}")


def log_parameters(params, level=logging.INFO):
    """
    Logs experiment parameters.

    Args:
        params: Dictionary of experiment parameters
        level: Logging level
    """
    log_dict("Experiment Parameters", params, level)
