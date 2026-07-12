import unittest
from pathlib import Path

import pandas as pd

from MLOps.app.api.model_service import ModelInputError, PredictionService


DATA_PLATFORM_DIR = Path(__file__).resolve().parents[2]


class PredictionServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = PredictionService(
            DATA_PLATFORM_DIR / "Model" / "artifacts" / "logistic_regression_abt.pkl"
        )
        cls.service.load()
        abt = pd.read_csv(
            DATA_PLATFORM_DIR / "Dados" / "abt.csv",
            nrows=1,
        )
        cls.features = abt.drop(columns=["sk_id_curr", "target"]).iloc[0].to_dict()

    def test_prediction_is_valid(self) -> None:
        score, predicted_class = self.service.predict(self.features)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        self.assertIn(predicted_class, {0, 1})

    def test_missing_features_are_rejected(self) -> None:
        with self.assertRaises(ModelInputError):
            self.service.predict({"age": 35})


if __name__ == "__main__":
    unittest.main()
