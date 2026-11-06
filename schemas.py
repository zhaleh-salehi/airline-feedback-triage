from datetime import datetime
from pydantic import field_validator
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Input: Receive one tweet
class CommentCreate(BaseModel):
    tweet_id: str = Field(min_length=1)
    airline_sentiment: str | None = None
    airline: str | None = None
    name: str | None = None
    text: str = Field(min_length=1)
    tweet_created: datetime | None = None

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Tweet text cannot be empty or whitespace")
        return value


# Output: Tweet with AI results
class CommentOut(CommentCreate):
    id: int

    classification: Literal[
        "complaint",
        "compliment",
        "request"
    ]

    sentiment: Literal[
        "positive",
        "negative",
        "neutral"
    ]

    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Output: LLM analysis history
class AnalysisOut(BaseModel):
    id: int
    comment_id: int

    model: str
    prompt_version: str

    classification: Literal[
        "complaint",
        "compliment",
        "request"
    ]

    sentiment: Literal[
        "positive",
        "negative",
        "neutral"
    ]

    tokens_used: int | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)