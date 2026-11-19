import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import SentimentModel


@pytest.fixture(scope="session")
def sentiment_model() -> SentimentModel:
    return SentimentModel.load()


@pytest.fixture(scope="session")
def client(sentiment_model: SentimentModel) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client
