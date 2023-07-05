import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X,
                rooms_per_household,
                population_per_household,
                bedrooms_per_room,
            ]

        else:
            return np.c_[X, rooms_per_household, population_per_household]


def build_pipeline():
    num_pipeline = num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )
    num_attribs = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ]
    )

    forest_reg = RandomForestRegressor(max_features=4, n_estimators=30)

    final_pipeline = Pipeline(
        [("preparation", full_pipeline), ("random_reg", forest_reg)]
    )
    return final_pipeline


def train_model(input_path, output_path):
    """
    This creates a pickle file for trained model.

    Parameters
    ----------
    input_path : str
        This is file path of training data.
    output_path: str
        This is directory path for pickle file.

    Returns
    ------
        This function doesn't return anything
    """
    logger.info(f"Reading data from {input_path}...")
    data = pd.read_csv(input_path)
    logger.info("Data loaded successfully")
    train_data = data.drop("median_house_path", axis=1)
    train_labels = data["median_house_path"].copy()
    pipeline = build_pipeline()
    pipeline.fit(train_data, train_labels)
    joblib.dump(pipeline, output_path)
    logger.info(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--input-folder",
        type=str,
        default="./data",
        help="Folder path for input datasets",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="./artifacts",
        help="Folder path for output models",
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

    # create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # train model
    input_path = os.path.join(args.input_folder, "train.csv")
    output_path = os.path.join(args.output_folder, "model.pickle")
    train_model(input_path, output_path)


if __name__ == "__main__":
    main()
