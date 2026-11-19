from prometheus_client import Counter, Histogram

from app.schemas import SENTIMENT_LABELS

PREDICTIONS_TOTAL = Counter(
    "sentiment_predictions_total",
    "Total predictions by sentiment label",
    ["label"],
)

INPUT_TEXT_LENGTH = Histogram(
    "sentiment_input_text_length_chars",
    "Length of input text passed to /predict (characters)",
    buckets=(8, 32, 64, 128, 256, 512, 1024, 2000),
)


def record_prediction(label: str, text_length: int) -> None:
    if label not in SENTIMENT_LABELS:
        label = "neutral"
    PREDICTIONS_TOTAL.labels(label=label).inc()
    INPUT_TEXT_LENGTH.observe(text_length)
