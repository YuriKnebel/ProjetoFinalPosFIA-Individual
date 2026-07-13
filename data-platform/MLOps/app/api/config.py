import os
from dataclasses import dataclass
from pathlib import Path


DATA_PLATFORM_DIR = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(
        os.getenv(
            "MODEL_PATH",
            str(DATA_PLATFORM_DIR / "Model" / "artifacts" / "lightgbm_abt.pkl"),
        )
    )
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://airflow:airflow@postgres:5432/data",
    )
    approve_max_score: float = float(os.getenv("CREDIT_APPROVE_MAX_SCORE", "0.50"))
    manual_review_max_score: float = float(
        os.getenv("CREDIT_MANUAL_REVIEW_MAX_SCORE", "0.60")
    )
    policy_version: str = os.getenv("CREDIT_POLICY_VERSION", "demo-v1")

    def validate(self) -> None:
        if not 0 <= self.approve_max_score < self.manual_review_max_score <= 1:
            raise ValueError(
                "Os limiares devem respeitar: "
                "0 <= CREDIT_APPROVE_MAX_SCORE < "
                "CREDIT_MANUAL_REVIEW_MAX_SCORE <= 1."
            )


settings = Settings()
