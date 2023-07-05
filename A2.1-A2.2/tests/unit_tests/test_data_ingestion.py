import os
import unittest

from myhousingprice.ingest_data import (
    data_preparation_function,
    fetch_housing_data,
    load_housing_data,
)


class TestIngestData(unittest.TestCase):
    def test_download_dataset(self):
        DOWNLOAD_ROOT = (
            "https://raw.githubusercontent.com/ageron/handson-ml/master/"
        )
        HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
        data_folder = "./tests/data"
        fetch_housing_data(HOUSING_URL, data_folder)
        df = load_housing_data(data_folder)
        result = data_preparation_function(df)
        self.assertTrue(
            os.path.exists(os.path.join(data_folder, "housing.csv"))
        )
        self.assertCountEqual(list(result.keys()), ["train", "test"])


if __name__ == "__main__":
    unittest.main()
