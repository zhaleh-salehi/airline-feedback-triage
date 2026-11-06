from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def simulate_ai_failure(text):
    raise RuntimeError("Simulated OpenAI failure")


def test_openai_failure():
    tweet = {
        "tweet_id": "test_ai_failure_001",
        "airline_sentiment": "neutral",
        "airline": "United",
        "name": "test_user",
        "text": "Can you help me change my flight?",
        "tweet_created": "2026-11-06T08:00:00Z"
    }

    with patch(
        "main.analyzer.analyze_text",
        side_effect=simulate_ai_failure
    ):
        response = client.post("/comments", json=tweet)

    assert response.status_code == 502
    assert response.json()["detail"] == "AI analysis failed"

    print("AI failure handling: PASSED")


if __name__ == "__main__":
    test_openai_failure()