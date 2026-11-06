from unittest.mock import patch
from sqlalchemy.exc import OperationalError
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import models
from database import Base, get_db
from main import app


# Temporary SQLite database for testing
test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

Base.metadata.create_all(bind=test_engine)

TestingSessionLocal = sessionmaker(
    bind=test_engine,
    autoflush=False,
    autocommit=False,
)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


def test_database_rollback():
    tweet = {
        "tweet_id": "test_rollback_001",
        "airline_sentiment": "negative",
        "airline": "United",
        "name": "test_user",
        "text": "My flight was delayed!",
        "tweet_created": "2026-11-06T08:00:00Z",
    }

    fake_ai_result = {
        "classification": "complaint",
        "sentiment": "negative",
        "tokens_used": 100,
    }

    # Simulate a database failure when inserting an analysis
    def fail_analysis_insert(conn, cursor, statement, parameters, context, executemany):
        if statement.strip().lower().startswith("insert into analyses"):
            raise OperationalError(statement, parameters, Exception("Simulated database failure"))

    event.listen(test_engine, "before_cursor_execute", fail_analysis_insert)

    try:
        with patch("main.analyzer.analyze_text", return_value=fake_ai_result):
            # TestClient normally raises unhandled exceptions
            with TestClient(app, raise_server_exceptions=False) as test_client:
                response = test_client.post("/comments", json=tweet)

        assert response.status_code == 500

        with TestingSessionLocal() as db:
            comment = (
                db.query(models.Comment)
                .filter_by(tweet_id="test_rollback_001")
                .first()
            )

            analysis_count = db.query(models.Analysis).count()

            assert comment is None
            assert analysis_count == 0

        print("Database rollback: PASSED")

    finally:
        event.remove(test_engine, "before_cursor_execute", fail_analysis_insert)
        app.dependency_overrides.clear()
        test_engine.dispose()


if __name__ == "__main__":
    test_database_rollback()