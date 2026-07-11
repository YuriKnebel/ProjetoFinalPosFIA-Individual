import json
import pickle
import unittest
from pathlib import Path


DATA_PLATFORM_DIR = Path(__file__).resolve().parents[2]


class ConfigurationTest(unittest.TestCase):
    def test_item_c_required_structure_exists(self) -> None:
        required_paths = (
            "Dados/README.md",
            "DataPipeline/data_sanitization.py",
            "DataPipeline/abt_transform.py",
            "DataPipeline/exp_analysis.ipynb",
            "DataPipeline/config_pipeline.json",
            "Model/train.py",
            "Model/config_model.json",
            "Model/evaluation.ipynb",
            "MLOps/app/api/main.py",
            "MLOps/app/frontend/app.py",
            "airflow/dags/pipeline_orchestration.py",
            "requirements.txt",
        )
        missing = [
            path for path in required_paths if not (DATA_PLATFORM_DIR / path).is_file()
        ]
        self.assertEqual(missing, [])
        self.assertTrue((DATA_PLATFORM_DIR.parent / "README.md").is_file())

    def test_required_sections_exist_in_both_configurations(self) -> None:
        for relative_path in (
            "DataPipeline/config_pipeline.json",
            "Model/config_model.json",
        ):
            config = json.loads(
                (DATA_PLATFORM_DIR / relative_path).read_text(encoding="utf-8")
            )
            self.assertTrue({"metadata", "variables", "parameters"} <= config.keys())

    def test_model_features_match_persisted_artifact(self) -> None:
        config = json.loads(
            (DATA_PLATFORM_DIR / "Model/config_model.json").read_text(
                encoding="utf-8"
            )
        )
        with (
            DATA_PLATFORM_DIR / "Model/artifacts/logistic_regression_abt.pkl"
        ).open("rb") as file:
            artifact = pickle.load(file)
        self.assertEqual(
            config["variables"]["input_features"],
            artifact["input_features"],
        )


if __name__ == "__main__":
    unittest.main()
