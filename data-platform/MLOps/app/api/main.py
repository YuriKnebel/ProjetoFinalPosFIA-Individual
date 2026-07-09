from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from .config import settings
from .credit_policy import CreditPolicy
from .feature_service import CustomerFeatureService, CustomerNotFoundError
from .model_service import ModelInputError, PredictionService
from .schemas import (
    FeaturePredictionRequest,
    HealthResponse,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate()

    prediction_service = PredictionService(settings.model_path)
    prediction_service.load()

    database_engine = create_engine(settings.database_url, pool_pre_ping=True)
    feature_service = CustomerFeatureService(database_engine)
    credit_policy = CreditPolicy(
        approve_max_score=settings.approve_max_score,
        manual_review_max_score=settings.manual_review_max_score,
        version=settings.policy_version,
    )

    app.state.prediction_service = prediction_service
    app.state.feature_service = feature_service
    app.state.credit_policy = credit_policy

    yield

    database_engine.dispose()


app = FastAPI(
    title="API de Risco de Crédito",
    description=(
        "Expõe o modelo por features prontas ou por cliente armazenado no banco. "
        "A recomendação final é produzida por uma política separada do modelo."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service: PredictionService = request.app.state.prediction_service
    return HealthResponse(
        status="ok",
        model_loaded=service.is_loaded,
        model_path=str(service.model_path),
    )


@app.get("/model/features", response_model=list[str])
def model_features(request: Request) -> list[str]:
    service: PredictionService = request.app.state.prediction_service
    return service.expected_features


@app.post("/predict/features", response_model=PredictionResponse)
def predict_from_features(
    payload: FeaturePredictionRequest, request: Request
) -> PredictionResponse:
    return _predict(
        features=payload.features,
        source="provided_features",
        request=request,
    )


@app.post("/predict/customer/{customer_id}", response_model=PredictionResponse)
def predict_from_database(customer_id: int, request: Request) -> PredictionResponse:
    feature_service: CustomerFeatureService = request.app.state.feature_service

    try:
        features = feature_service.build(customer_id)
    except CustomerNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=503,
            detail="Não foi possível consultar as fontes de dados do cliente.",
        ) from error

    return _predict(
        features=features,
        source="database",
        customer_id=customer_id,
        request=request,
    )


def _predict(
    features: dict,
    source: str,
    request: Request,
    customer_id: int | None = None,
) -> PredictionResponse:
    prediction_service: PredictionService = request.app.state.prediction_service
    credit_policy: CreditPolicy = request.app.state.credit_policy

    try:
        risk_score, predicted_class = prediction_service.predict(features)
    except ModelInputError as error:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Features obrigatórias ausentes.",
                "missing_features": error.missing_features,
            },
        ) from error

    policy_decision = credit_policy.evaluate(risk_score)

    return PredictionResponse(
        source=source,
        customer_id=customer_id,
        risk_score=risk_score,
        predicted_class=predicted_class,
        model_decision_threshold=prediction_service.decision_threshold,
        policy=asdict(policy_decision),
    )
