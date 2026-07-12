from dataclasses import dataclass
from typing import Literal


Recommendation = Literal["approve", "manual_review", "reject"]


@dataclass(frozen=True)
class PolicyDecision:
    recommendation: Recommendation
    reason: str
    policy_version: str
    approve_max_score: float
    manual_review_max_score: float


class CreditPolicy:
    """Transforma score de risco em recomendação de negócio configurável."""

    def __init__(
        self,
        approve_max_score: float,
        manual_review_max_score: float,
        version: str,
    ) -> None:
        if not 0 <= approve_max_score < manual_review_max_score <= 1:
            raise ValueError("Limiares inválidos para a política de crédito.")

        self.approve_max_score = approve_max_score
        self.manual_review_max_score = manual_review_max_score
        self.version = version

    def evaluate(self, risk_score: float) -> PolicyDecision:
        if not 0 <= risk_score <= 1:
            raise ValueError("O score de risco deve estar entre 0 e 1.")

        if risk_score < self.approve_max_score:
            recommendation: Recommendation = "approve"
            reason = "Score abaixo do limite de aprovação da política."
        elif risk_score < self.manual_review_max_score:
            recommendation = "manual_review"
            reason = "Score na faixa intermediária; requer análise humana."
        else:
            recommendation = "reject"
            reason = "Score acima do limite máximo aceito pela política."

        return PolicyDecision(
            recommendation=recommendation,
            reason=reason,
            policy_version=self.version,
            approve_max_score=self.approve_max_score,
            manual_review_max_score=self.manual_review_max_score,
        )
