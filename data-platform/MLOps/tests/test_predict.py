import unittest
from pathlib import Path

import pandas as pd

from Model.predict import load_artifact, predict_score


DATA_PLATFORM_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = DATA_PLATFORM_DIR / "Model" / "artifacts" / "lightgbm_abt.pkl"


class PredictScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = load_artifact(ARTIFACT_PATH)

    def test_predicts_sample_row(self) -> None:
        categories = self.artifact.get("categories", {})
        row = {
            name: categories[name][0] if name in categories else 0.0
            for name in self.artifact["features"]
        }
        features = pd.DataFrame([row])

        result = predict_score(features, self.artifact)

        self.assertGreaterEqual(result["risk_score"], 0)
        self.assertLessEqual(result["risk_score"], 1)
        self.assertIn(result["predicted_class"], {0, 1})
        self.assertEqual(result["decision_threshold"], 0.5)
        self.assertIn(result["decision"], {"APROVAR_CREDITO", "NEGAR_CREDITO"})


if __name__ == "__main__":
    unittest.main()
