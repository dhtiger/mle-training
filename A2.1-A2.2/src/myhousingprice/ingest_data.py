import argparse
import logging
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("data", "raw")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    This Function downloads raw data from source.

    Parameters
    ----------
    housing_url : str
        This is raw data source url.
    housing_path : str
        This is directory path to store the raw data files.

    Returns
    -------
        This is non returning fucntion.
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    logger.info(f"Downloading data from {housing_url}...")
    urllib.request.urlretrieve(housing_url, tgz_path)
    logger.info(f"Data downloaded successfully to {housing_path}")
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """
    This is loading function from csv to pandas.

    Parameters
    ----------
    housing_path : str
        This is path of the raw dataset folder.

    Returns
    -------
    Pandas.DataFrame
        This is DataFrame of the housing csv file.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def data_preparation_function(housing):
    """
    This function genrates test and validation datasets in a dict format.

    Parameters
    ----------
    housing : pd.DataFrame
        This is input raw data that has to be split.

    Returns:
    Dict
        This is dict with train and test with respective Dataframes.
    """
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    logger.debug("Train test data split successfully")

    return {"train": strat_train_set, "test": strat_test_set}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="Path of folder to save processed files",
        default="./data/processed",
    )
    parser.add_argument(
        "--log-level",
        help="Mention logging level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--log-path",
        type=str,
        help="Mention log file path can be None eg: logs/<file_name> ",
        default=None,
    )
    parser.add_argument(
        "--no-console-log",
        help="Disable console logging",
        action="store_true",
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), None)
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
            logging.Formatter(("%(levelname)s - %(message)s"))
        )
        logger.addHandler(console_handler)

    os.makedirs(args.path, exist_ok=True)
    fetch_housing_data()
    raw_data = load_housing_data()
    processed_data = data_preparation_function(raw_data)
    processed_data["train"].to_csv(args.path + "train.csv")
    processed_data["test"].to_csv(args.path + "test.csv")
    logging.info(f"Train and Test data saved at {args.path}")
