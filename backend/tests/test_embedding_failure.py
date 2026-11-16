from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def simulate_embedding_failure(text):
    raise RuntimeError("Simulated embedding failure")


def simulate_ai_success(text):
    return {
        "classification": ["complaint", "request"],
        "sentiment": "negative",
        "tokens_used": 100
    }


def test_embedding_failure():

    tweet = {
        "tweet_id": f"test_embedding_failure_{uuid4().hex}",
        "airline_sentiment": "negative",
        "airline": "United",
        "name": "test_user",
        "text": "My flight was cancelled. Can you help me?",
        "tweet_created": "2026-11-06T08:00:00Z"
    }

    with patch(
        "backend.main.analyzer.analyze_text",
        side_effect=simulate_ai_success
    ), patch(
        "backend.main.embedding.generate_embedding",
        side_effect=simulate_embedding_failure
    ):

        response = client.post(
            "/comments",
            json=tweet
        )

    assert response.status_code == 502

    assert (
        response.json()["detail"]
        == "Embedding generation failed"
    )

    print("Embedding failure handling: PASSED")


if __name__ == "__main__":
    test_embedding_failure()