import logging
import os

import pandas as pd

from mle_training import ingest_data

HOUSING_PATH = os.path.join("../data/raw", "housing")


def test_ingest_data_raw():
    assert type(ingest_data.load_housing_data(HOUSING_PATH)) == pd.DataFrame


def test_ingest_data_processed():
    HOUSING_PATH = os.path.join("../data/processed", "housing")
    pth = os.listdir(HOUSING_PATH)
    n_files = len(pth)
    assert n_files == 4


def test_train():

    HOUSING_PATH = os.path.join("../artifacts", "housing")

    pth = os.listdir(HOUSING_PATH)
    n_files = len(pth)
    assert n_files == 3
