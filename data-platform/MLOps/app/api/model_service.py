import pickle
from pathlib import Path
from typing import Any

import pandas as pd


class ModelInputError(ValueError):
    def __init__(self, missing_features: list[str]) -> None:
        self.missing_features = missing_features
        super().__init__(f"Features ausentes: {', '.join(missing_features)}")


class PredictionService:
    """Carrega o artefato e oferece uma interface única de scoring."""

    REQUIRED_ARTIFACT_KEYS = {
        "model",
        "decision_threshold",
        "input_features",
        "metrics",
    }

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.artifact: dict[str, Any] | None = None

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")

        with self.model_path.open("rb") as file:
            artifact = pickle.load(file)
        missing_keys = self.REQUIRED_ARTIFACT_KEYS.difference(artifact)
        if missing_keys:
            raise ValueError(
                f"Artefato inválido. Chaves ausentes: {sorted(missing_keys)}"
            )

        self.artifact = artifact

    @property
    def is_loaded(self) -> bool:
        return self.artifact is not None

    @property
    def expected_features(self) -> list[str]:
        self._ensure_loaded()
        return list(self.artifact["input_features"])

    @property
    def decision_threshold(self) -> float:
        self._ensure_loaded()
        return float(self.artifact["decision_threshold"])

    def predict(self, features: dict[str, Any]) -> tuple[float, int]:
        self._ensure_loaded()

        missing_features = sorted(set(self.expected_features).difference(features))
        if missing_features:
            raise ModelInputError(missing_features)

        # Reindex garante ordem idêntica à do treinamento e ignora campos extras.
        customer = pd.DataFrame([features]).reindex(columns=self.expected_features)
        risk_score = float(self.artifact["model"].predict_proba(customer)[0, 1])
        predicted_class = int(risk_score >= self.decision_threshold)
        return risk_score, predicted_class

    def _ensure_loaded(self) -> None:
        if self.artifact is None:
            raise RuntimeError("O modelo ainda não foi carregado.")
