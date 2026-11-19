from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_TEXT_CHARS = 2000
SENTIMENT_LABELS = ("negative", "neutral", "positive")


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def non_empty_after_strip(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be empty")
        return stripped


class PredictResponse(BaseModel):
    label: Literal["negative", "neutral", "positive"]
    confidence: float = Field(ge=0.0, le=1.0)
    scores: dict[Literal["negative", "neutral", "positive"], float]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_id: str
    model_loaded: bool
