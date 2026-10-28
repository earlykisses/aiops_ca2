from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Twitter Sentiment Analysis API"
    )


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json={
            "text": "I really love this game!",
            "entity": "Gaming",
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["sentiment"] in {
        "Positive",
        "Negative",
        "Neutral",
        "Irrelevant",
    }

    assert 0 <= result["confidence"] <= 1


def test_empty_text():
    response = client.post(
        "/predict",
        json={
            "text": "",
            "entity": "Gaming",
        },
    )

    assert response.status_code == 422


def test_missing_text():
    response = client.post(
        "/predict",
        json={
            "entity": "Gaming",
        },
    )

    assert response.status_code == 422