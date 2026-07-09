"""Executa inferência local com o artefato treinado.

Exemplos (a partir de ``data-platform``):

    python Model/predict.py --input Dados/abt.csv --row 0
    python Model/predict.py --input cliente.json

O CSV pode conter ``sk_id_curr`` e ``target``; essas colunas são ignoradas.
O JSON deve ser um objeto com as features ou conter a chave ``features``.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent
CONFIG_PATH = MODEL_DIR / "config_model.json"
NON_FEATURE_COLUMNS = {"sk_id_curr", "target"}


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Carrega variáveis, parâmetros e metadados documentados do modelo."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"metadata", "variables", "parameters"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Configuração inválida; seções ausentes: {sorted(missing)}")
    return config


def default_model_path() -> Path:
    configured = load_config()["metadata"]["artifact"]
    return MODEL_DIR.parent / configured


def load_artifact(model_path: Path | None = None) -> dict[str, Any]:
    """Carrega e valida o pacote produzido pelo notebook de treinamento."""
    model_path = model_path or default_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    with model_path.open("rb") as file:
        artifact = pickle.load(file)
    required = {"model", "decision_threshold", "input_features"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"Artefato inválido; chaves ausentes: {sorted(missing)}")
    return artifact


def load_features(input_path: Path, row: int = 0) -> dict[str, Any]:
    """Lê as features de um JSON ou de uma linha de CSV."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Entrada não encontrada: {input_path}")

    if input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        features = payload.get("features", payload)
        if not isinstance(features, dict):
            raise ValueError("O JSON deve ser um objeto de features.")
        return features

    if input_path.suffix.lower() == ".csv":
        frame = pd.read_csv(input_path, skiprows=range(1, row + 1), nrows=1)
        if frame.empty:
            raise IndexError(f"A linha {row} não existe no CSV.")
        return {
            key: value
            for key, value in frame.iloc[0].to_dict().items()
            if key not in NON_FEATURE_COLUMNS
        }

    raise ValueError("Formato não suportado. Use um arquivo .json ou .csv.")


def predict(features: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    """Calcula score e classe, preservando a ordem usada no treinamento."""
    expected = list(artifact["input_features"])
    missing = sorted(set(expected).difference(features))
    if missing:
        raise ValueError(f"Features ausentes: {', '.join(missing)}")

    customer = pd.DataFrame([features]).reindex(columns=expected)
    score = float(artifact["model"].predict_proba(customer)[0, 1])
    threshold = float(artifact["decision_threshold"])
    return {
        "risk_score": score,
        "predicted_class": int(score >= threshold),
        "decision_threshold": threshold,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predição local de risco de crédito")
    parser.add_argument("--input", required=True, type=Path, help="Arquivo JSON ou CSV")
    parser.add_argument("--row", type=int, default=0, help="Linha do CSV (inicia em zero)")
    parser.add_argument("--model", type=Path, default=default_model_path())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.row < 0:
        raise ValueError("--row deve ser maior ou igual a zero.")
    result = predict(load_features(args.input, args.row), load_artifact(args.model))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
