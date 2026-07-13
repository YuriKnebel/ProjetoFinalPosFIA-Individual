import asyncio
from contextlib import asynccontextmanager
from contextlib import suppress
from dataclasses import asdict
import json
import logging

from fastapi import FastAPI, HTTPException, Request
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from .config import settings
from .credit_policy import CreditPolicy
from .feature_service import CustomerFeatureService, CustomerNotFoundError
from .model_service import ModelInputError, PredictionService
from .schemas import (
    CustomerFeaturesResponse,
    FeaturePredictionRequest,
    HealthResponse,
    PredictionResponse,
)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate()

    prediction_service = PredictionService(settings.model_path)
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
    app.state.model_load_error = None

    model_load_task = asyncio.create_task(
        _load_model_with_retry(
            app,
            prediction_service,
            settings.model_load_retry_seconds,
        )
    )

    try:
        yield
    finally:
        model_load_task.cancel()
        with suppress(asyncio.CancelledError):
            await model_load_task
        database_engine.dispose()


async def _load_model_with_retry(
    app: FastAPI,
    service: PredictionService,
    retry_seconds: float,
) -> None:
    while not service.is_loaded:
        try:
            await asyncio.to_thread(service.load)
        except Exception as error:
            app.state.model_load_error = str(error)
            logger.error(
                "Falha ao carregar o modelo %s: %s. Nova tentativa em %.1f segundos.",
                service.model_path,
                error,
                retry_seconds,
            )
            await asyncio.sleep(retry_seconds)
        else:
            app.state.model_load_error = None
            logger.info("Modelo carregado com sucesso: %s", service.model_path)


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
    if not service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unavailable",
                "model_loaded": False,
                "model_path": str(service.model_path),
                "message": "O artefato do modelo ainda não foi carregado.",
                "last_error": request.app.state.model_load_error,
            },
        )

    return HealthResponse(
        status="ok",
        model_loaded=service.is_loaded,
        model_path=str(service.model_path),
    )


@app.get("/model/features", response_model=list[str])
def model_features(request: Request) -> list[str]:
    service: PredictionService = request.app.state.prediction_service
    _ensure_model_loaded(service)
    return service.expected_features


@app.get("/customers/{customer_id}/features", response_model=CustomerFeaturesResponse)
def customer_features(customer_id: int, request: Request) -> CustomerFeaturesResponse:
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

    return CustomerFeaturesResponse(customer_id=customer_id, features=features)


@app.post("/predict/features", response_model=PredictionResponse)
def predict_from_features(
    payload: FeaturePredictionRequest, request: Request
) -> PredictionResponse:
    _log_request_json(
        "POST /predict/features",
        {"features": payload.features},
    )
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

    _log_request_json(
        f"POST /predict/customer/{customer_id}",
        {"customer_id": customer_id, "features": features},
    )
    return _predict(
        features=features,
        source="database",
        customer_id=customer_id,
        request=request,
    )

def _ensure_model_loaded(service: PredictionService) -> None:
    if service.is_loaded:
        return

    raise HTTPException(
        status_code=503,
        detail="O modelo ainda não está disponível.",
    )

def _log_request_json(endpoint: str, payload: dict) -> None:
    print(
        f"Request JSON {endpoint}: "
        f"{json.dumps(payload, ensure_ascii=False, default=str)}",
        flush=True,
    )


def _predict(
    features: dict,
    source: str,
    request: Request,
    customer_id: int | None = None,
) -> PredictionResponse:
    prediction_service: PredictionService = request.app.state.prediction_service
    credit_policy: CreditPolicy = request.app.state.credit_policy

    _ensure_model_loaded(prediction_service)

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
        policy=asdict(policy_decision),
    )
