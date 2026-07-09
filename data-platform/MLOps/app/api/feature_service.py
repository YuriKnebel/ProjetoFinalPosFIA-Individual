from typing import Any

import pandas as pd
from sqlalchemy import Engine, text


class CustomerNotFoundError(LookupError):
    pass


APPLICATION_QUERY = text(
    """
    SELECT
        sk_id_curr,
        ext_source_1,
        ext_source_2,
        ext_source_3,
        region_rating_client_w_city,
        days_last_phone_change,
        days_id_publish,
        days_registration,
        reg_city_not_work_city,
        reg_city_not_live_city,
        live_city_not_work_city,
        flag_own_car,
        own_car_age,
        def_60_cnt_social_circle,
        amt_req_credit_bureau_year,
        cnt_children,
        cnt_fam_members,
        amt_income_total,
        amt_credit,
        amt_annuity,
        occupation_type,
        organization_type,
        name_income_type,
        name_education_type,
        code_gender,
        days_birth,
        days_employed
    FROM application_train
    WHERE sk_id_curr = :customer_id
    """
)


PREVIOUS_APPLICATION_QUERY = text(
    """
    SELECT
        COUNT(sk_id_prev) AS prev_contract_count,
        SUM(CASE WHEN name_contract_status = 'Refused' THEN 1 ELSE 0 END)
            AS prev_refused_count
    FROM previous_application
    WHERE sk_id_curr = :customer_id
    """
)


BUREAU_QUERY = text(
    """
    SELECT
        COUNT(sk_id_bureau) AS bureau_credit_count,
        AVG(days_credit) AS bureau_avg_days_credit,
        MAX(days_credit) AS bureau_last_days_credit,
        SUM(CASE WHEN TRIM(credit_active) = 'Active' THEN 1 ELSE 0 END)
            AS bureau_active_count,
        SUM(CASE WHEN TRIM(credit_active) = 'Closed' THEN 1 ELSE 0 END)
            AS bureau_closed_count,
        SUM(amt_credit_sum) AS bureau_total_credit,
        SUM(amt_credit_sum_debt) AS bureau_total_debt,
        SUM(amt_credit_sum_overdue) AS bureau_total_overdue,
        SUM(CASE WHEN credit_day_overdue > 0 THEN 1 ELSE 0 END)
            AS bureau_overdue_count
    FROM bureau
    WHERE sk_id_curr = :customer_id
    """
)


RELEVANT_ORGANIZATIONS_QUERY = text(
    """
    SELECT organization_type AS value
    FROM application_train
    GROUP BY organization_type
    HAVING COUNT(*) >= 500
    """
)


RELEVANT_INCOME_TYPES_QUERY = text(
    """
    SELECT name_income_type AS value
    FROM application_train
    GROUP BY name_income_type
    HAVING COUNT(*) >= 500
    """
)


ZERO_FILLED_NUMERIC_FEATURES = {
    "reg_city_not_work_city",
    "reg_city_not_live_city",
    "live_city_not_work_city",
    "own_car_age",
    "def_60_cnt_social_circle",
    "amt_req_credit_bureau_year",
    "cnt_children",
}


class CustomerFeatureService:
    """Reproduz para um cliente as features usadas na construção da ABT."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._relevant_organizations: set[str] | None = None
        self._relevant_income_types: set[str] | None = None

    def build(self, customer_id: int) -> dict[str, Any]:
        self._load_category_mappings()

        with self.engine.connect() as connection:
            application = connection.execute(
                APPLICATION_QUERY, {"customer_id": customer_id}
            ).mappings().first()

            if application is None:
                raise CustomerNotFoundError(
                    f"Cliente {customer_id} não encontrado em application_train."
                )

            previous = connection.execute(
                PREVIOUS_APPLICATION_QUERY, {"customer_id": customer_id}
            ).mappings().one()
            bureau = connection.execute(
                BUREAU_QUERY, {"customer_id": customer_id}
            ).mappings().one()

        return self._transform(application, previous, bureau)

    def _load_category_mappings(self) -> None:
        if self._relevant_organizations is not None:
            return

        with self.engine.connect() as connection:
            self._relevant_organizations = {
                row.value
                for row in connection.execute(RELEVANT_ORGANIZATIONS_QUERY)
                if row.value is not None
            }
            self._relevant_income_types = {
                row.value
                for row in connection.execute(RELEVANT_INCOME_TYPES_QUERY)
                if row.value is not None
            }

    def _transform(
        self,
        application: dict[str, Any],
        previous: dict[str, Any],
        bureau: dict[str, Any],
    ) -> dict[str, Any]:
        features = {
            key: self._python_value(value)
            for key, value in application.items()
            if key not in {"sk_id_curr", "days_birth", "days_employed", "flag_own_car"}
        }

        for feature in ZERO_FILLED_NUMERIC_FEATURES:
            if features.get(feature) is None:
                features[feature] = 0

        if features.get("amt_income_total") == 0:
            features["amt_income_total"] = None
        if features.get("amt_income_total") is None:
            features["amt_income_total"] = 0.001

        ext_values = [
            features.get("ext_source_1"),
            features.get("ext_source_2"),
            features.get("ext_source_3"),
        ]
        ext_values = [value for value in ext_values if value is not None]
        features["ext_source_mean"] = (
            sum(ext_values) / len(ext_values) if ext_values else None
        )
        if features.get("ext_source_mean") is None:
            features["ext_source_mean"] = features.get("ext_source_2")

        flag_own_car = self._clean_category(application.get("flag_own_car"), "N")
        features["has_car"] = int(flag_own_car == "Y")

        features["occupation_type"] = self._clean_category(
            features.get("occupation_type"), "Unknown"
        )
        features["organization_type"] = self._condense_category(
            features.get("organization_type"), self._relevant_organizations
        )
        features["name_income_type"] = self._condense_category(
            features.get("name_income_type"), self._relevant_income_types
        )
        features["name_education_type"] = self._clean_category(
            features.get("name_education_type"), "Unknown"
        )

        gender = self._clean_category(features.get("code_gender"), "Unknown")
        features["code_gender"] = "Unknown" if gender == "XNA" else gender

        days_birth = self._python_value(application.get("days_birth"))
        days_employed = self._python_value(application.get("days_employed"))
        features["age"] = abs(days_birth) / 365.25 if days_birth is not None else None
        features["days_employed_anom"] = int(days_employed == 365243)
        if days_employed == 365243:
            days_employed = 0
        features["years_employed"] = (
            abs(days_employed) / 365.25 if days_employed is not None else 0
        )
        features["fe_credit_income_percent"] = (
            features["amt_credit"] / features["amt_income_total"]
            if features.get("amt_credit") is not None
            else None
        )
        features["fe_annuity_income_percent"] = (
            features["amt_annuity"] / features["amt_income_total"]
            if features.get("amt_annuity") is not None
            else None
        )

        previous_count = int(previous["prev_contract_count"] or 0)
        previous_refused = int(previous["prev_refused_count"] or 0)
        features["prev_refused_rate"] = (
            previous_refused / previous_count if previous_count else None
        )
        features["has_prev_app"] = int(previous_count > 0)

        bureau_count = int(bureau["bureau_credit_count"] or 0)
        bureau_active = int(bureau["bureau_active_count"] or 0)
        bureau_closed = int(bureau["bureau_closed_count"] or 0)
        features["bureau_avg_days_credit"] = self._python_value(
            bureau["bureau_avg_days_credit"]
        )
        features["bureau_last_days_credit"] = self._python_value(
            bureau["bureau_last_days_credit"]
        )
        features["bureau_active_rate"] = (
            bureau_active / bureau_count if bureau_count else None
        )
        features["bureau_active_count"] = bureau_active if bureau_count else None
        features["bureau_closed_rate"] = (
            bureau_closed / bureau_count if bureau_count else None
        )
        bureau_total_credit = self._python_value(bureau["bureau_total_credit"]) or 0
        bureau_total_debt = self._python_value(bureau["bureau_total_debt"]) or 0
        features["bureau_debt_credit_ratio"] = (
            bureau_total_debt / bureau_total_credit if bureau_total_credit else 0
        )
        features["bureau_debt_credit_ratio"] = max(
            -1,
            min(1, features["bureau_debt_credit_ratio"]),
        )
        features["bureau_overdue_count"] = int(bureau["bureau_overdue_count"] or 0)
        features["has_bureau"] = int(bureau_count > 0)

        return features

    @staticmethod
    def _clean_category(value: Any, missing_value: str) -> str:
        if value is None or pd.isna(value):
            return missing_value
        return str(value).strip()

    @classmethod
    def _condense_category(
        cls, value: Any, relevant_values: set[str] | None
    ) -> str:
        cleaned = cls._clean_category(value, "Other_low_freq")
        return cleaned if cleaned in (relevant_values or set()) else "Other_low_freq"

    @staticmethod
    def _python_value(value: Any) -> Any:
        if value is None or pd.isna(value):
            return None
        if hasattr(value, "item"):
            return value.item()
        return value
