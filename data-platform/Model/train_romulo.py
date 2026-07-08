"""Treina o modelo de risco de crédito usando ``config_model.json``.

Uso, a partir de ``data-platform``::

    .venv/bin/python Model/train.py
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_DIR = Path(__file__).resolve().parent
DATA_PLATFORM_DIR = MODEL_DIR.parent
DEFAULT_CONFIG_PATH = MODEL_DIR / "config_model.json"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    missing = {"metadata", "variables", "parameters"}.difference(config)
    if missing:
        raise ValueError(f"Configuração incompleta; seções ausentes: {sorted(missing)}")
    return config


def project_path(configured_path: str) -> Path:
    return DATA_PLATFORM_DIR / configured_path


def load_training_data(config: dict[str, Any], sample_size: int | None = None):
    path = project_path(config["metadata"]["training_dataset"])
    frame = pd.read_csv(path, nrows=sample_size)
    variables = config["variables"]
    required = set(variables["input_features"]) | {variables["target"]}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"A ABT não contém as colunas configuradas: {missing}")

    X = frame[variables["input_features"]].replace([np.inf, -np.inf], np.nan)
    y = frame[variables["target"]].astype(int)
    return X, y


def build_estimator(config: dict[str, Any], X_train: pd.DataFrame) -> Pipeline:
    preprocessing = config["parameters"]["preprocessing"]
    classifier = config["parameters"]["classifier"]
    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = X_train.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=preprocessing["numeric_imputer"])),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=preprocessing["categorical_imputer"])),
        ("onehot", OneHotEncoder(
            handle_unknown=preprocessing["one_hot_handle_unknown"],
            min_frequency=preprocessing["one_hot_min_frequency"],
        )),
    ])
    transformer = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])
    return Pipeline([
        ("preprocessor", transformer),
        ("classifier", LogisticRegression(
            solver=classifier["solver"],
            max_iter=classifier["max_iter"],
            random_state=classifier["random_state"],
        )),
    ])


def train(
    config: dict[str, Any],
    sample_size: int | None = None,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    X, y = load_training_data(config, sample_size)
    split = config["parameters"]["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split["test_size"],
        stratify=y if split["stratify"] else None,
        random_state=split["random_state"],
    )

    search_config = config["parameters"]["hyperparameter_search"]
    estimator = build_estimator(config, X_train)
    search = GridSearchCV(
        estimator=estimator,
        param_grid={
            "classifier__C": search_config["classifier__C"],
            "classifier__class_weight": search_config["classifier__class_weight"],
        },
        scoring=search_config["scoring"],
        cv=StratifiedKFold(
            n_splits=search_config["cv_folds"],
            shuffle=True,
            random_state=split["random_state"],
        ),
        n_jobs=search_config["n_jobs"] if n_jobs is None else n_jobs,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    score = search.best_estimator_.predict_proba(X_test)[:, 1]
    threshold = config["parameters"]["inference"]["decision_threshold"]
    prediction = (score >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, score)),
        "average_precision": float(average_precision_score(y_test, score)),
        "cv_roc_auc": float(search.best_score_),
    }
    print(json.dumps({"best_params": search.best_params_, "metrics": metrics}, indent=2))
    print(classification_report(y_test, prediction, digits=4))

    return {
        "model": search.best_estimator_,
        "decision_threshold": threshold,
        "input_features": config["variables"]["input_features"],
        "metrics": metrics,
        "best_params": search.best_params_,
        "cv_folds": search_config["cv_folds"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_version": config["metadata"]["version"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--sample-size", type=int, help="Amostra para smoke test")
    parser.add_argument("--n-jobs", type=int, help="Sobrescreve o paralelismo configurado")
    parser.add_argument("--output", type=Path, help="Sobrescreve o caminho do artefato")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    artifact = train(config, args.sample_size, args.n_jobs)
    output = args.output or project_path(config["metadata"]["artifact"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(artifact, file)
    print(f"Artefato salvo em: {output.resolve()}")


if __name__ == "__main__":
    main()
