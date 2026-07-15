import pickle
import unittest
from pathlib import Path

from MLOps.app.api.model_service import ModelInputError, PredictionService


DATA_PLATFORM_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = DATA_PLATFORM_DIR / "Model" / "artifacts" / "lightgbm_abt.pkl"


def build_sample_features(artifact_path: Path) -> dict:
    """Monta um cliente de teste a partir do contrato salvo no proprio artefato.

    Evita depender de CSVs externos (nao versionados): as categoricas usam a
    primeira categoria persistida e as numericas recebem zero.
    """
    with artifact_path.open("rb") as file:
        artifact = pickle.load(file)
    categories = artifact.get("categories", {})
    return {
        name: categories[name][0] if name in categories else 0.0
        for name in artifact["features"]
    }


class PredictionServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = PredictionService(ARTIFACT_PATH)
        cls.service.load()
        cls.features = build_sample_features(ARTIFACT_PATH)

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
