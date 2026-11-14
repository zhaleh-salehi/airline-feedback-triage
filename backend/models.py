from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    CheckConstraint,
)

from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship

from backend.database import Base


# ==========================================
# Table 1: Comments
# ==========================================

class Comment(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, index=True)

    tweet_id = Column(String, unique=True, nullable=True)

    # Original dataset information
    airline_sentiment = Column(String, nullable=True)
    airline = Column(String, nullable=True)
    name = Column(String, nullable=True)
    text = Column(Text, nullable=False)
    tweet_created = Column(DateTime(timezone=True), nullable=True)

    # Vector embedding of the tweet
    embedding = Column(Vector(1536), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # One comment can have multiple analyses
    analyses = relationship(
        "Analysis",
        back_populates="comment",
        cascade="all, delete-orphan",
    )


# ==========================================
# Table 2: Analyses
# ==========================================

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)

    comment_id = Column(
        Integer,
        ForeignKey("comments.id"),
        nullable=False,
    )

    model = Column(String, nullable=False)
    prompt_version = Column(String, nullable=False)

    # Multi-label classification
    classification = Column(ARRAY(String), nullable=False)

    # Single sentiment
    sentiment = Column(String, nullable=False)

    tokens_used = Column(Integer, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Sentiment validation
    __table_args__ = (
        CheckConstraint(
            "sentiment IN ('positive', 'negative', 'neutral')",
            name="check_analysis_sentiment",
        ),
    )

    # Each analysis belongs to one comment
    comment = relationship(
        "Comment",
        back_populates="analyses",
    )