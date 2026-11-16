import re
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from backend import models
from backend.database import Base, engine, get_db
from backend.main import app


# ==========================================
# 1. Connect to TEST Database
# ==========================================

test_url = engine.url.set(
    database="airline_feedback_test"
)

test_engine = create_engine(test_url)

# Create tables only in the test database
Base.metadata.create_all(bind=test_engine)

TestingSessionLocal = sessionmaker(
    bind=test_engine,
    autocommit=False,
    autoflush=False
)


# ==========================================
# 2. Override Database Connection
# ==========================================

def override_get_db():
    db = TestingSessionLocal()

    try:
        yield db

    finally:
        db.close()


# ==========================================
# 3. Mock OpenAI Responses
# ==========================================

def fake_ai_analysis(text):
    return {
        "classification": [
            "complaint",
            "request"
        ],
        "sentiment": "negative",
        "tokens_used": 100
    }


def fake_embedding(text):
    return [0.0] * 1536


# ==========================================
# 4. Simulate Database Failure
# ==========================================

def simulate_database_failure(
    conn,
    cursor,
    statement,
    parameters,
    context,
    executemany
):
    # Trigger failure only when inserting
    # a record into the analyses table

    if re.match(
        r'^\s*INSERT\s+INTO\s+(?:"?\w+"?\.)?"?analyses"?\b',
        statement,
        re.IGNORECASE
    ):
        raise OperationalError(
            statement,
            parameters,
            Exception("Simulated database failure")
        )


# ==========================================
# 5. Rollback Test
# ==========================================

def test_database_rollback():

    tweet_id = f"test_rollback_{uuid4().hex}"

    tweet = {
        "tweet_id": tweet_id,
        "airline_sentiment": "negative",
        "airline": "United",
        "name": "test_user",
        "text": "My flight was delayed!",
        "tweet_created": "2026-11-06T08:00:00Z"
    }

    # Make sure the database is still empty
    with TestingSessionLocal() as db:
        assert db.query(models.Comment).count() == 0
        assert db.query(models.Analysis).count() == 0

    # Use the test database for this request
    app.dependency_overrides[get_db] = override_get_db

    event.listen(
        test_engine,
        "before_cursor_execute",
        simulate_database_failure
    )

    try:

        # Mock both OpenAI API calls
        with patch(
            "backend.main.analyzer.analyze_text",
            side_effect=fake_ai_analysis
        ), patch(
            "backend.main.embedding.generate_embedding",
            side_effect=fake_embedding
        ):

            with TestClient(
                app,
                raise_server_exceptions=False
            ) as client:

                response = client.post(
                    "/comments",
                    json=tweet
                )

        # Database error should return HTTP 500
        assert response.status_code == 500, (
            f"Unexpected response: "
            f"{response.status_code} {response.text}"
        )

        assert (
            response.json()["detail"]
            == "Database error while saving tweet"
        )

        # Check that no incomplete data was saved
        with TestingSessionLocal() as db:

            comment = (
                db.query(models.Comment)
                .filter_by(tweet_id=tweet_id)
                .first()
            )

            comment_count = db.query(
                models.Comment
            ).count()

            analysis_count = db.query(
                models.Analysis
            ).count()

            assert comment is None
            assert comment_count == 0
            assert analysis_count == 0

        print("Database rollback: PASSED")

    finally:

        # Remove simulated database failure
        event.remove(
            test_engine,
            "before_cursor_execute",
            simulate_database_failure
        )

        # Restore original database dependency
        app.dependency_overrides.pop(get_db, None)


if __name__ == "__main__":

    try:
        test_database_rollback()

    finally:
        test_engine.dispose()