from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from backend import analyzer, embedding, models, schemas
from backend.database import engine, get_db


# ==========================================
# Database Initialization
# ==========================================

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Airline Feedback Triage System"
)


# ==========================================
# Root Endpoint
# ==========================================

@app.get("/")
def root():
    return {"status": "ok"}


# ==========================================
# POST /comments
# Receive, Analyze, Embed and Save ONE Tweet
# ==========================================

@app.post(
    "/comments",
    response_model=schemas.CommentOut,
    status_code=201
)
def create_comment(
    comment: schemas.CommentCreate,
    db: Session = Depends(get_db)
):

    # Step 1: Check duplicate tweet
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

    # Also reject the exact same text from the same known username,
    # even if the new tweet has a different tweet_id.
    username = (comment.name or "").strip()
    if username:
        try:
            existing_same_text = (
                db.query(models.Comment.id)
                .filter(
                    func.lower(func.btrim(models.Comment.name)) == username.lower(),
                    models.Comment.text == comment.text,
                )
                .first()
            )
        except SQLAlchemyError as exc:
            raise HTTPException(
                status_code=500,
                detail="Database error while checking duplicates"
            ) from exc

        if existing_same_text:
            raise HTTPException(
                status_code=409,
                detail="Duplicate tweet: same username and exact text"
            )

    # Step 2: AI Classification + Sentiment
    try:
        result = analyzer.analyze_text(comment.text)

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="AI analysis failed"
        ) from exc

    # Step 3: Generate Embedding
    try:
        tweet_embedding = embedding.generate_embedding(
            comment.text
        )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Embedding generation failed"
        ) from exc

    # Step 4: Save Tweet + Analysis
    try:

        # Store tweet and embedding
        db_comment = models.Comment(
            **comment.model_dump(),
            embedding=tweet_embedding
        )

        db.add(db_comment)
        db.flush()

        # Store multi-label classification
        db_analysis = models.Analysis(
            comment_id=db_comment.id,
            model=analyzer.MODEL,
            prompt_version=analyzer.PROMPT_VERSION,
            classification=result["classification"],
            sentiment=result["sentiment"],
            tokens_used=result["tokens_used"]
        )

        db.add(db_analysis)

        # Commit both records together
        db.commit()

        # Reload tweet and its analysis
        return (
            db.query(models.Comment)
            .options(selectinload(models.Comment.analyses))
            .filter(models.Comment.id == db_comment.id)
            .one()
        )

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


# ==========================================
# GET /comments
# List stored tweets
# ==========================================

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
        .options(selectinload(models.Comment.analyses))
        .order_by(models.Comment.id)
        .offset(skip)
        .limit(limit)
        .all()
    )


# ==========================================
# GET /comments/search
# Semantic Search using pgvector
# ==========================================

@app.get(
    "/comments/search",
    response_model=list[schemas.CommentOut]
)
def semantic_search(
    query: str = Query(..., min_length=2),
    limit: int = Query(default=5, ge=1, le=50),
    db: Session = Depends(get_db)
):

    # Step 1: Validate search query
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty"
        )

    # Step 2: Generate query embedding
    try:
        query_embedding = embedding.generate_embedding(
            query.strip()
        )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Search embedding generation failed"
        ) from exc

    # Step 3: Calculate semantic distance
    distance = models.Comment.embedding.cosine_distance(
        query_embedding
    )

    # Step 4: Search stored tweets
    try:
        results = (
            db.query(models.Comment)
            .options(
                selectinload(models.Comment.analyses)
            )
            .filter(
                models.Comment.embedding.isnot(None)
            )
            .order_by(distance)
            .limit(limit)
            .all()
        )

        return results

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Semantic search database error"
        ) from exc


# ==========================================
# POST /comments/check-duplicates
# Read-only exact duplicate detection (no embeddings or AI)
# ==========================================

@app.post(
    "/comments/check-duplicates",
    response_model=schemas.DuplicateCheckResponse,
)
def check_exact_duplicates(
    request: schemas.DuplicateCheckRequest,
    db: Session = Depends(get_db),
):
    # A missing username must never match another anonymous tweet.
    username = (request.name or "").strip()
    if not username:
        return schemas.DuplicateCheckResponse(
            checked=False,
            potential_duplicate=False,
            reason="Username is missing; exact duplicate check skipped",
        )

    # Same username (ignoring capitalization/outer spaces) and EXACT text.
    # No semantic similarity or threshold is involved.
    try:
        previous_tweets = (
            db.query(models.Comment)
            .filter(
                func.lower(func.btrim(models.Comment.name)) == username.lower(),
                models.Comment.text == request.text,
            )
            .order_by(models.Comment.id)
            .limit(5)
            .all()
        )
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500, detail="Duplicate check database error"
        ) from exc

    matches = [
        schemas.PotentialDuplicateOut(
            comment_id=previous.id,
            tweet_id=previous.tweet_id,
            name=previous.name,
            text=previous.text,
            similarity=1.0,
        )
        for previous in previous_tweets
    ]
    return schemas.DuplicateCheckResponse(
        checked=True,
        potential_duplicate=bool(matches),
        matches=matches,
    )


# ==========================================
# GET /comments/{comment_id}
# Get one stored tweet
# ==========================================

@app.get(
    "/comments/{comment_id}",
    response_model=schemas.CommentOut
)
def get_comment(
    comment_id: int,
    db: Session = Depends(get_db)
):

    comment = (
        db.query(models.Comment)
        .options(selectinload(models.Comment.analyses))
        .filter(models.Comment.id == comment_id)
        .first()
    )

    if comment is None:
        raise HTTPException(
            status_code=404,
            detail="Comment not found"
        )

    return comment


# ==========================================
# GET /analyses
# List stored analyses
# ==========================================

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

# ==========================================
# PATCH /comments/{comment_id}/classification
# Manual Classification Override
# ==========================================

@app.patch(
    "/comments/{comment_id}/classification",
    response_model=schemas.AnalysisOut
)
def update_classification(
    comment_id: int,
    update: schemas.ClassificationUpdate,
    db: Session = Depends(get_db)
):

    # Find the selected tweet
    comment = db.get(models.Comment, comment_id)

    if comment is None:
        raise HTTPException(
            status_code=404,
            detail="Comment not found"
        )

    # Find the latest existing analysis
    analysis = (
        db.query(models.Analysis)
        .filter(models.Analysis.comment_id == comment_id)
        .order_by(models.Analysis.id.desc())
        .first()
    )

    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )

    # Human decision overrides AI classification and sentiment
    analysis.classification = update.classification
    analysis.sentiment = update.sentiment

    try:
        db.commit()
        db.refresh(analysis)

    except SQLAlchemyError as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Failed to update classification"
        ) from exc

    return analysis
# ==========================================
# DELETE /comments/{comment_id}
# Permanently Remove Tweet
# ==========================================

@app.delete("/comments/{comment_id}")
def delete_comment(
    comment_id: int,
    db: Session = Depends(get_db)
):

    # Find the selected tweet
    comment = db.get(models.Comment, comment_id)

    if comment is None:
        raise HTTPException(
            status_code=404,
            detail="Comment not found"
        )

    # Delete tweet and related analyses
    try:
        db.delete(comment)
        db.commit()

    except SQLAlchemyError as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Failed to delete tweet"
        ) from exc

    return {
        "message": "Tweet deleted successfully",
        "deleted_comment_id": comment_id
    }