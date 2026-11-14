from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

import analyzer
import models
import schemas

from database import engine, get_db


models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Airline Feedback Triage System")


@app.get("/")
def root():
    return {"status": "ok"}


# Receive, analyze and save ONE tweet
@app.post(
    "/comments",
    response_model=schemas.CommentOut,
    status_code=201
)
def create_comment(
    comment: schemas.CommentCreate,
    db: Session = Depends(get_db)
):

    # Step 1: Check for duplicate tweet
    existing = (
        db.query(models.Comment)
        .filter_by(tweet_id=comment.tweet_id)
        .first()
    )

    if existing:
        raise HTTPException(
            status_code=409,
            detail="Tweet already exists"
        )

    # Step 2: Analyze the tweet with AI
    try:
        result = analyzer.analyze_text(comment.text)

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="AI analysis failed"
        ) from exc

    # Step 3: Save tweet and analysis together
    try:
        db_comment = models.Comment(
            **comment.model_dump(),
            classification=result["classification"],
            sentiment=result["sentiment"]
        )

        db.add(db_comment)
        db.flush()

        db_analysis = models.Analysis(
            comment_id=db_comment.id,
            model=analyzer.MODEL,
            prompt_version=analyzer.PROMPT_VERSION,
            classification=result["classification"],
            sentiment=result["sentiment"],
            tokens_used=result["tokens_used"]
        )

        db.add(db_analysis)

        db.commit()
        db.refresh(db_comment)

        return db_comment

    except IntegrityError as exc:
        db.rollback()

        raise HTTPException(
            status_code=409,
            detail="Tweet already exists or violates a database constraint"
        ) from exc

    except SQLAlchemyError as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Database error while saving tweet"
        ) from exc


# Get all processed tweets
@app.get(
    "/comments",
    response_model=list[schemas.CommentOut]
)
def list_comments(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db)
):

    return (
        db.query(models.Comment)
        .order_by(models.Comment.id)
        .offset(skip)
        .limit(limit)
        .all()
    )


# Get one processed tweet
@app.get(
    "/comments/{comment_id}",
    response_model=schemas.CommentOut
)
def get_comment(
    comment_id: int,
    db: Session = Depends(get_db)
):

    comment = db.get(models.Comment, comment_id)

    if comment is None:
        raise HTTPException(
            status_code=404,
            detail="Comment not found"
        )

    return comment


# Get analysis history
@app.get(
    "/analyses",
    response_model=list[schemas.AnalysisOut]
)
def list_analyses(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db)
):

    return (
        db.query(models.Analysis)
        .order_by(models.Analysis.id)
        .offset(skip)
        .limit(limit)
        .all()
    )