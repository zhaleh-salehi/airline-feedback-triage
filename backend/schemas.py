from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ==========================================
# Input: Receive One Tweet
# ==========================================

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


# ==========================================
# Output: AI Analysis
# ==========================================

class AnalysisOut(BaseModel):
    id: int
    comment_id: int

    model: str
    prompt_version: str

    classification: list[
        Literal["complaint", "compliment", "request"]
    ]

    sentiment: Literal["positive", "negative", "neutral"]

    tokens_used: int | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# Output: Tweet + Analysis
# ==========================================

class CommentOut(CommentCreate):
    id: int
    created_at: datetime

    analyses: list[AnalysisOut] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
# ==========================================
# Input: Manual Classification & Sentiment Override
# ==========================================

class ClassificationUpdate(BaseModel):
    classification: list[
        Literal["complaint", "compliment", "request"]
    ] = Field(min_length=1, max_length=3)

    sentiment: Literal[
        "positive", "negative", "neutral"
    ]

    @field_validator("classification")
    @classmethod
    def validate_classification(cls, value):
        if len(value) != len(set(value)):
            raise ValueError(
                "Duplicate classification labels are not allowed."
            )
        return value