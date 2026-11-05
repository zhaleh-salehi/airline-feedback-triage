from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    CheckConstraint,
)
from sqlalchemy.orm import relationship

from database import Base


class Comment(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, index=True)
    tweet_id = Column(String, unique=True, nullable=True)

    airline_sentiment = Column(String, nullable=True)
    airline = Column(String, nullable=True)
    name = Column(String, nullable=True)
    text = Column(Text, nullable=False)
    tweet_created = Column(DateTime(timezone=True), nullable=True)

    classification = Column(String, nullable=False)
    sentiment = Column(String, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "classification IN ('complaint', 'compliment', 'request')",
            name="check_comment_classification",
        ),
        CheckConstraint(
            "sentiment IN ('positive', 'negative', 'neutral')",
            name="check_comment_sentiment",
        ),
    )

    analyses = relationship(
        "Analysis",
        back_populates="comment",
        cascade="all, delete-orphan",
    )


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

    classification = Column(String, nullable=False)
    sentiment = Column(String, nullable=False)

    tokens_used = Column(Integer, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "classification IN ('complaint', 'compliment', 'request')",
            name="check_analysis_classification",
        ),
        CheckConstraint(
            "sentiment IN ('positive', 'negative', 'neutral')",
            name="check_analysis_sentiment",
        ),
    )

    comment = relationship(
        "Comment",
        back_populates="analyses",
    )