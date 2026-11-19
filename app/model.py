from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_TOKEN_LENGTH = 512


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    scores: dict[str, float]


class SentimentModel:
    """Wraps a Hugging Face sequence-classification model (loaded once per process)."""

    _instance: ClassVar[SentimentModel | None] = None

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()

    @classmethod
    def load(cls, model_id: str = MODEL_ID) -> SentimentModel:
        if cls._instance is None:
            cls._instance = cls(model_id=model_id)
        return cls._instance

    def predict(self, text: str) -> PredictionResult:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
        )
        with torch.no_grad():
            logits = self.model(**encoded).logits[0]

        probabilities = torch.softmax(logits, dim=-1).tolist()
        labels = [self.model.config.id2label[i].lower() for i in range(len(probabilities))]
        scores = dict(zip(labels, probabilities, strict=True))
        best_idx = int(torch.argmax(logits))
        label = labels[best_idx]
        confidence = float(probabilities[best_idx])
        return PredictionResult(label=label, confidence=confidence, scores=scores)
