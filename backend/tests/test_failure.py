from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def simulate_ai_failure(text):
    raise RuntimeError("Simulated OpenAI failure")


def test_openai_failure():

    tweet = {
        "tweet_id": f"test_ai_failure_{uuid4().hex}",
        "airline_sentiment": "neutral",
        "airline": "United",
        "name": "test_user",
        "text": "Can you help me change my flight?",
        "tweet_created": "2026-11-06T08:00:00Z"
    }

    with patch(
        "backend.main.analyzer.analyze_text",
        side_effect=simulate_ai_failure
    ):
        response = client.post("/comments", json=tweet)

    assert response.status_code == 502

    assert (
        response.json()["detail"]
        == "AI analysis failed"
    )

    print("AI failure handling: PASSED")


if __name__ == "__main__":
    test_openai_failure()