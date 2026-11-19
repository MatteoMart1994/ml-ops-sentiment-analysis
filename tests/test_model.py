import csv
from pathlib import Path

import pytest

from app.model import SentimentModel
from app.schemas import SENTIMENT_LABELS

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_sentiment.csv"

# The published model scores ~0.72 accuracy on the full TweetEval test split;
# a much lower value on this sample means the wrong model or a broken pipeline.
MIN_SAMPLE_ACCURACY = 0.5


@pytest.mark.parametrize(
    "text,expected",
    [
        ("This is absolutely wonderful, best day ever!", "positive"),
        ("I hate this, it is terrible and disappointing.", "negative"),
    ],
)
def test_model_clear_sentiments(sentiment_model: SentimentModel, text: str, expected: str) -> None:
    result = sentiment_model.predict(text)
    assert result.label == expected
    assert result.confidence > 0.5
    assert set(result.scores) == set(SENTIMENT_LABELS)


def test_model_on_tweeteval_sample(sentiment_model: SentimentModel) -> None:
    with DATASET_PATH.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows, "sample dataset is empty"

    hits = 0
    for row in rows:
        pred = sentiment_model.predict(row["text"])
        assert pred.label in SENTIMENT_LABELS
        hits += pred.label == row["label"]

    assert hits / len(rows) >= MIN_SAMPLE_ACCURACY
