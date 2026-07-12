import unittest
from pathlib import Path

from Model.predict import load_artifact, load_features, predict


DATA_PLATFORM_DIR = Path(__file__).resolve().parents[2]


class PredictScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = load_artifact(
            DATA_PLATFORM_DIR / "Model" / "artifacts" / "logistic_regression_abt.pkl"
        )

    def test_predicts_csv_row(self) -> None:
        features = load_features(DATA_PLATFORM_DIR / "Dados" / "abt.csv")
        result = predict(features, self.artifact)

        self.assertGreaterEqual(result["risk_score"], 0)
        self.assertLessEqual(result["risk_score"], 1)
        self.assertIn(result["predicted_class"], {0, 1})
        self.assertEqual(result["decision_threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
