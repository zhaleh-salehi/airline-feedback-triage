from fastapi import Depends, FastAPI, HTTPException, Query
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