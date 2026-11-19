from app.schemas import MAX_TEXT_CHARS


def test_health_returns_ok(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert "cardiffnlp/twitter-roberta-base-sentiment-latest" in payload["model_id"]


def test_predict_response_schema(client) -> None:
    response = client.post("/predict", json={"text": "I love this product!"})
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"label", "confidence", "scores"}
    assert data["label"] in {"positive", "negative", "neutral"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert set(data["scores"].keys()) == {"positive", "negative", "neutral"}
    assert abs(sum(data["scores"].values()) - 1.0) < 0.05


def test_predict_rejects_empty_text(client) -> None:
    for body in ({"text": ""}, {"text": "   \n\t"}):
        response = client.post("/predict", json=body)
        assert response.status_code == 422


def test_predict_rejects_too_long_text(client) -> None:
    long_text = "x" * (MAX_TEXT_CHARS + 1)
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code == 422


def test_metrics_endpoint_exposes_custom_counters(client) -> None:
    client.post("/predict", json={"text": "This is wonderful!"})
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    body = metrics.text
    assert "sentiment_predictions_total" in body
    assert "sentiment_input_text_length_chars" in body
    assert "http_requests_total" in body
