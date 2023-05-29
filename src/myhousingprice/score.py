import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def score_model(model_path, data_path, output_path):
    """ "
    This function prints root mean square error score.

    Parameters
    ----------
    model_path : str
        This is path of model pickle file.
    data_path : str
        This is path of test dataset.
    output_path : str
        This is path of directory for predictions to a csv

    Returns
    -------
        This function doesnt't return but print rmse score
    """
    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
    logger.info(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    test_data = data.drop("median_house_value", axis=1)
    test_labels = data["median_house_value"].copy()
    logger.info("Data loaded successfully")
    final_predictions = model.predict(test_data)
    logger.info("Scoring model...")
    final_mse = mean_squared_error(test_labels, final_predictions)
    final_rmse = np.sqrt(final_mse)
    logger.info(f"Model scored {final_rmse}")
    print(f"RMSE: {final_rmse}")
    # save output to file
    final_predictions.to_csv(output_path)
    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Score the model")
    parser.add_argument(
        "--model-folder",
        type=str,
        default="./artifacts",
        help="Folder path for trained models",
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        default="./data",
        help="Folder path for input data",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="./artifacts",
        help="Folder path for output data",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path for log file (if not, logs will be written to console)",
    )
    parser.add_argument(
        "--no-console-log", action="store_true", help="Disable console logging"
    )
    args = parser.parse_args()

    # configure logging
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError("Invalid log level: %s" % args.log_level)
    logger.setLevel(log_level)

    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    if not args.no_console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(levelname)s - %(message)s")
        )
        logger.addHandler(console_handler)

    # score model
    model_path = os.path.join(args.model_folder, "model.pickle")
    data_path = os.path.join(args.data_folder, "test.csv")
    output_path = os.path.join(args.output_folder, "results.csv")
    score_model(model_path, data_path, output_path)


if __name__ == "__main__":
    main()
