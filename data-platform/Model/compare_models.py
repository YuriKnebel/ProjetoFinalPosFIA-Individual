"""Compara modelos de classificação usando a mesma ABT e o mesmo split.

Uso, a partir de ``data-platform``::

    .venv/bin/python Model/compare_models.py

Para um teste rápido::

    .venv/bin/python Model/compare_models.py --sample-size 5000 --n-jobs 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from train import DEFAULT_CONFIG_PATH, load_config, load_training_data, project_path


DEFAULT_OUTPUT = "Model/artifacts/model_comparison.csv"


def build_preprocessor(config: dict[str, Any], X_train: pd.DataFrame) -> ColumnTransformer:
    preprocessing = config["parameters"]["preprocessing"]
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
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])


def candidate_models(config: dict[str, Any], n_jobs: int | None) -> dict[str, Any]:
    classifier_config = config["parameters"]["classifier"]
    random_state = classifier_config["random_state"]
    effective_n_jobs = -1 if n_jobs is None else n_jobs

    return {
        "logistic_regression": LogisticRegression(
            solver=classifier_config["solver"],
            max_iter=classifier_config["max_iter"],
            class_weight="balanced",
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=100,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=100,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=effective_n_jobs,
        ),
    }


def evaluate_models(
    config: dict[str, Any],
    sample_size: int | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    X, y = load_training_data(config, sample_size)
    split = config["parameters"]["split"]
    threshold = config["parameters"]["inference"]["decision_threshold"]
    search_config = config["parameters"]["hyperparameter_search"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split["test_size"],
        stratify=y if split["stratify"] else None,
        random_state=split["random_state"],
    )

    cv = StratifiedKFold(
        n_splits=search_config["cv_folds"],
        shuffle=True,
        random_state=split["random_state"],
    )
    preprocessor = build_preprocessor(config, X_train)
    rows: list[dict[str, Any]] = []

    for model_name, classifier in candidate_models(config, n_jobs).items():
        estimator = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])
        started_at = time.perf_counter()
        cv_scores = cross_val_score(
            estimator,
            X_train,
            y_train,
            scoring="roc_auc",
            cv=cv,
            n_jobs=n_jobs,
        )
        estimator.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started_at

        score = estimator.predict_proba(X_test)[:, 1]
        prediction = (score >= threshold).astype(int)
        rows.append({
            "model": model_name,
            "roc_auc": float(roc_auc_score(y_test, score)),
            "accuracy": float(accuracy_score(y_test, prediction)),
            "precision": float(precision_score(y_test, prediction, zero_division=0)),
            "average_precision": float(average_precision_score(y_test, score)),
            "cv_roc_auc_mean": float(np.mean(cv_scores)),
            "cv_roc_auc_std": float(np.std(cv_scores)),
            "fit_seconds": round(fit_seconds, 2),
            "decision_threshold": threshold,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["roc_auc", "precision", "accuracy"], ascending=False)
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--sample-size", type=int, help="Amostra para smoke test")
    parser.add_argument("--n-jobs", type=int, help="Paralelismo usado na validação cruzada")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="CSV de saída")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    comparison = evaluate_models(config, args.sample_size, args.n_jobs)
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False)

    print(json.dumps(comparison.to_dict(orient="records"), indent=2))
    print(f"Comparação salva em: {output.resolve()}")


if __name__ == "__main__":
    main()
