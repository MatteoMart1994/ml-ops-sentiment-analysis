from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator

from app.metrics import record_prediction
from app.model import MODEL_ID, SentimentModel
from app.schemas import HealthResponse, PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.sentiment_model = SentimentModel.load()
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="Sentiment Inference",
        version="1.0.0",
        lifespan=lifespan,
    )

    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
    ).instrument(
        application
    ).expose(application, include_in_schema=False)

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        model: SentimentModel | None = getattr(request.app.state, "sentiment_model", None)
        loaded = model is not None
        status = "ok" if loaded else "degraded"
        return HealthResponse(status=status, model_id=MODEL_ID, model_loaded=loaded)

    @application.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        model: SentimentModel = request.app.state.sentiment_model
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        try:
            result = model.predict(body.text)
        except Exception as exc:  # pragma: no cover — unexpected inference failure
            raise HTTPException(status_code=500, detail="inference failed") from exc

        record_prediction(result.label, len(body.text))
        return PredictResponse(
            label=result.label,  # type: ignore[arg-type]
            confidence=result.confidence,
            scores=result.scores,  # type: ignore[arg-type]
        )

    return application


app = create_app()
