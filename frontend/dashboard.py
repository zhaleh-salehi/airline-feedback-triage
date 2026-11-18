import streamlit as st
import pandas as pd
import requests

from collections import Counter
from datetime import datetime
from pathlib import Path

from sklearn.metrics import f1_score
from sqlalchemy import select, func

from backend.database import SessionLocal
from backend.models import Comment, Analysis


# ==========================================
# Dashboard Configuration
# ==========================================

st.set_page_config(
    page_title="Airline Feedback Dashboard",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Airline Feedback Dashboard")

st.write("Airline Feedback Triage System")


# ==========================================
# API Configuration
# ==========================================

API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 90


# ==========================================
# Database Functions
# ==========================================

def get_total_tweets():

    with SessionLocal() as db:

        total = db.scalar(
            select(func.count(Comment.id))
        )

        return total or 0


def get_latest_analyses():

    with SessionLocal() as db:

        latest_analysis = (
            select(
                Analysis.comment_id,
                func.max(Analysis.id).label("latest_id")
            )
            .group_by(Analysis.comment_id)
            .subquery()
        )

        results = db.execute(
            select(
                Comment.tweet_id,
                Comment.text,
                Analysis.classification,
                Analysis.sentiment
            )
            .join(
                latest_analysis,
                Comment.id == latest_analysis.c.comment_id
            )
            .join(
                Analysis,
                Analysis.id == latest_analysis.c.latest_id
            )
            .order_by(Comment.id.desc())
        ).all()

        return results


# ==========================================
# Semantic Search Functions
# ==========================================

def search_tweets(query, limit=5):
    """
    Search tweets by semantic similarity
    using the existing FastAPI endpoint.
    """

    response = requests.get(
        f"{API_BASE_URL}/comments/search",
        params={
            "query": query,
            "limit": limit
        },
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return response.json()


def prepare_search_results(results):
    """
    Convert API search results into
    a DataFrame for Streamlit.
    """

    rows = []

    for rank, tweet in enumerate(results, start=1):

        analyses = tweet.get("analyses") or []

        if analyses:

            latest_analysis = max(
                analyses,
                key=lambda item: item.get("id", 0)
            )

            classification = (
                latest_analysis.get("classification") or []
            )

            sentiment = latest_analysis.get(
                "sentiment", ""
            )

        else:

            classification = []
            sentiment = ""

        rows.append(
            {
                "Rank": rank,
                "Tweet ID": tweet.get("tweet_id", ""),
                "Tweet Text": tweet.get("text", ""),
                "Classification": ", ".join(
                    classification
                ),
                "Sentiment": sentiment
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Rank",
            "Tweet ID",
            "Tweet Text",
            "Classification",
            "Sentiment"
        ]
    )


# ==========================================
# CSV Processing Functions
# ==========================================

def normalize_tweet_date(value):
    """
    Convert CSV date values to ISO 8601.
    """

    value = str(value).strip()

    if not value:
        return None

    try:

        return datetime.strptime(
            value,
            "%Y-%m-%d %H:%M:%S %z"
        ).isoformat()

    except ValueError:
        pass

    try:

        return pd.to_datetime(
            value,
            utc=True,
            errors="raise"
        ).isoformat()

    except (ValueError, TypeError):

        raise ValueError(
            f"Invalid tweet_created date: {value}"
        )


def prepare_tweet(row):
    """
    Prepare one CSV row for POST /comments.
    """

    tweet_id = str(
        row.get("tweet_id", "")
    ).strip()

    text = str(
        row.get("text", "")
    ).strip()

    if not tweet_id:

        raise ValueError(
            "Tweet ID cannot be empty."
        )

    if not text:

        raise ValueError(
            "Tweet text cannot be empty."
        )

    def optional_value(column):

        value = str(
            row.get(column, "")
        ).strip()

        return value if value else None

    tweet = {
        "tweet_id": tweet_id,
        "airline_sentiment": optional_value(
            "airline_sentiment"
        ),
        "airline": optional_value(
            "airline"
        ),
        "name": optional_value(
            "name"
        ),
        "text": text,
        "tweet_created": normalize_tweet_date(
            row.get("tweet_created", "")
        )
    }

    return tweet


def process_csv_file(df):
    """
    Process CSV tweets through FastAPI.
    """

    total = len(df)

    successful = 0
    duplicates = 0
    failed = 0
    processed = 0

    errors = []

    progress_bar = st.progress(0)

    status_text = st.empty()

    stats_placeholder = st.empty()

    for index, row in df.iterrows():

        processed += 1

        status_text.info(
            f"Processing tweet {processed} of {total}..."
        )

        try:

            tweet = prepare_tweet(
                row.to_dict()
            )

            response = requests.post(
                f"{API_BASE_URL}/comments",
                json=tweet,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 201:

                successful += 1

            elif response.status_code == 409:

                duplicates += 1

            else:

                failed += 1

                errors.append(
                    {
                        "Row": processed,
                        "Tweet ID": tweet["tweet_id"],
                        "Error": (
                            f"HTTP {response.status_code}: "
                            f"{response.text[:300]}"
                        )
                    }
                )

                if response.status_code in (500, 502):

                    progress_bar.progress(
                        processed / total
                    )

                    break

        except ValueError as exc:

            failed += 1

            errors.append(
                {
                    "Row": processed,
                    "Tweet ID": str(
                        row.get("tweet_id", "")
                    ),
                    "Error": str(exc)
                }
            )

        except requests.exceptions.RequestException as exc:

            failed += 1

            errors.append(
                {
                    "Row": processed,
                    "Tweet ID": str(
                        row.get("tweet_id", "")
                    ),
                    "Error": str(exc)
                }
            )

            progress_bar.progress(
                processed / total
            )

            break

        progress = processed / total

        progress_bar.progress(progress)

        with stats_placeholder.container():

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Successful",
                    successful
                )

            with col2:

                st.metric(
                    "Duplicates",
                    duplicates
                )

            with col3:

                st.metric(
                    "Failed",
                    failed
                )

    completed = processed == total

    if completed:

        progress_bar.progress(1.0)

        status_text.success(
            "CSV processing completed!"
        )

    else:

        status_text.warning(
            "CSV processing stopped before all rows "
            "were processed."
        )

    return {
        "Total": total,
        "Processed": processed,
        "Successful": successful,
        "Duplicates": duplicates,
        "Failed": failed,
        "Completed": completed,
        "Errors": errors
    }


# ==========================================
# Semantic Search Interface
# ==========================================

st.header("🔎 Semantic Search")

st.write(
    "Search airline tweets by meaning, "
    "not just by matching exact keywords."
)

st.caption(
    "Example searches: lost baggage, flight delay, "
    "ticket refund, rude staff, excellent service."
)

with st.form("semantic_search_form"):

    search_query = st.text_input(
        "Enter a search query",
        placeholder="e.g. lost baggage"
    )

    search_limit = st.selectbox(
        "Maximum number of results",
        options=[3, 5, 10, 20],
        index=1
    )

    search_button = st.form_submit_button(
        "🔍 Search",
        type="primary"
    )


if search_button:

    if not search_query.strip():

        st.warning(
            "Please enter a search query."
        )

    elif len(search_query.strip()) < 2:

        st.warning(
            "Search query must contain at least "
            "2 characters."
        )

    else:

        try:

            with st.spinner(
                "Searching for semantically similar tweets..."
            ):

                search_results = search_tweets(
                    query=search_query.strip(),
                    limit=search_limit
                )

            st.session_state[
                "semantic_search_results"
            ] = search_results

            st.session_state[
                "semantic_search_query"
            ] = search_query.strip()

        except requests.exceptions.RequestException as exc:

            st.error(
                "Semantic Search failed. "
                "Please check that FastAPI is running."
            )

            st.exception(exc)

            st.session_state.pop(
                "semantic_search_results",
                None
            )


# ==========================================
# Semantic Search Results
# ==========================================

if "semantic_search_results" in st.session_state:

    results = st.session_state[
        "semantic_search_results"
    ]

    query_used = st.session_state.get(
        "semantic_search_query",
        ""
    )

    st.subheader("🔍 Search Results")

    st.write(
        f"Search query: **{query_used}**"
    )

    if results:

        st.success(
            f"Found {len(results)} nearest tweets."
        )

        search_df = prepare_search_results(
            results
        )

        st.dataframe(
            search_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn(
                    "Rank",
                    format="%d"
                ),
                "Tweet ID": st.column_config.TextColumn(
                    "Tweet ID"
                ),
                "Tweet Text": st.column_config.TextColumn(
                    "Tweet Text",
                    width="large"
                ),
                "Classification": st.column_config.TextColumn(
                    "Classification"
                ),
                "Sentiment": st.column_config.TextColumn(
                    "Sentiment"
                )
            }
        )

        st.caption(
            "Results are ranked by semantic similarity "
            "using OpenAI embeddings and PostgreSQL pgvector. "
            "The current search returns nearest matches "
            "without a minimum similarity threshold."
        )

    else:

        st.info(
            "No tweets with embeddings were found."
        )


st.divider()


# ==========================================
# CSV Upload Interface
# ==========================================

st.header("📂 CSV Upload & Batch Processing")

st.write(
    "Upload a CSV file containing airline tweets. "
    "The tweets will be analyzed and stored "
    "using the existing FastAPI backend."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    key="tweet_csv_upload"
)

if uploaded_file is not None:

    try:

        csv_df = pd.read_csv(
            uploaded_file,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig"
        )

        required_columns = [
            "tweet_id",
            "text"
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in csv_df.columns
        ]

        if missing_columns:

            st.error(
                "Missing required CSV columns: "
                + ", ".join(missing_columns)
            )

        elif csv_df.empty:

            st.warning(
                "The uploaded CSV file is empty."
            )

        else:

            st.success(
                f"CSV loaded successfully: "
                f"{len(csv_df)} tweets found."
            )

            st.subheader("CSV Preview")

            st.dataframe(
                csv_df.head(10),
                use_container_width=True,
                hide_index=True
            )

            if len(csv_df) > 20:

                st.info(
                    "For a quick project demonstration, "
                    "consider testing with approximately "
                    "10 tweets first. Larger files may "
                    "take longer and generate API costs."
                )

            start_processing = st.button(
                "🚀 Start Processing",
                type="primary"
            )

            if start_processing:

                st.session_state.pop(
                    "csv_processing_summary",
                    None
                )

                summary = process_csv_file(
                    csv_df
                )

                st.session_state[
                    "csv_processing_summary"
                ] = summary

                st.rerun()

    except Exception as exc:

        st.error(
            "Error while reading the CSV file."
        )

        st.exception(exc)


# ==========================================
# CSV Processing Summary
# ==========================================

if "csv_processing_summary" in st.session_state:

    summary = st.session_state[
        "csv_processing_summary"
    ]

    st.subheader("📊 Batch Processing Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Processed",
            summary["Processed"]
        )

    with col2:

        st.metric(
            "Successful",
            summary["Successful"]
        )

    with col3:

        st.metric(
            "Duplicates",
            summary["Duplicates"]
        )

    with col4:

        st.metric(
            "Failed",
            summary["Failed"]
        )

    if summary["Completed"]:

        st.success(
            "All CSV rows have been processed."
        )

    else:

        st.warning(
            "Processing stopped before completion."
        )

    if summary["Errors"]:

        st.subheader("Processing Errors")

        st.dataframe(
            pd.DataFrame(summary["Errors"]),
            use_container_width=True,
            hide_index=True
        )


st.divider()


# ==========================================
# Main Dashboard
# ==========================================

try:

    # ======================================
    # Total Tweets
    # ======================================

    total_tweets = get_total_tweets()

    st.metric(
        label="Total Analyzed Tweets",
        value=total_tweets
    )

    st.success(
        "Connected to PostgreSQL successfully!"
    )


    # ======================================
    # Load Latest Analyses
    # ======================================

    tweets = get_latest_analyses()


    # ======================================
    # Classification Distribution
    # ======================================

    classification_counts = Counter()

    for tweet in tweets:

        labels = tweet.classification

        if labels:

            classification_counts.update(
                set(labels)
            )

    st.subheader("Classification Distribution")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Complaints",
            classification_counts.get(
                "complaint", 0
            )
        )

    with col2:

        st.metric(
            "Compliments",
            classification_counts.get(
                "compliment", 0
            )
        )

    with col3:

        st.metric(
            "Requests",
            classification_counts.get(
                "request", 0
            )
        )


    # ======================================
    # Classification Chart
    # ======================================

    st.subheader("Classification Chart")

    classification_chart = pd.DataFrame(
        {
            "Classification": [
                "Complaint",
                "Compliment",
                "Request"
            ],
            "Count": [
                classification_counts.get(
                    "complaint", 0
                ),
                classification_counts.get(
                    "compliment", 0
                ),
                classification_counts.get(
                    "request", 0
                )
            ]
        }
    )

    st.bar_chart(
        classification_chart.set_index(
            "Classification"
        )
    )

    st.caption(
        "Multi-label classification: one tweet "
        "may belong to more than one category."
    )


    # ======================================
    # Sentiment Distribution
    # ======================================

    sentiment_counts = Counter()

    for tweet in tweets:

        if tweet.sentiment:

            sentiment_counts.update(
                [tweet.sentiment]
            )

    st.subheader("Sentiment Distribution")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Positive",
            sentiment_counts.get(
                "positive", 0
            )
        )

    with col2:

        st.metric(
            "Negative",
            sentiment_counts.get(
                "negative", 0
            )
        )

    with col3:

        st.metric(
            "Neutral",
            sentiment_counts.get(
                "neutral", 0
            )
        )


    # ======================================
    # Sentiment Chart
    # ======================================

    st.subheader("Sentiment Chart")

    sentiment_chart = pd.DataFrame(
        {
            "Sentiment": [
                "Positive",
                "Negative",
                "Neutral"
            ],
            "Count": [
                sentiment_counts.get(
                    "positive", 0
                ),
                sentiment_counts.get(
                    "negative", 0
                ),
                sentiment_counts.get(
                    "neutral", 0
                )
            ]
        }
    )

    st.bar_chart(
        sentiment_chart.set_index(
            "Sentiment"
        )
    )

    # ======================================
    # Analyzed Tweets & Classification Filter
    # ======================================

    st.subheader("Analyzed Tweets")

    st.write(
        "Explore analyzed airline tweets and filter "
        "them by their classification."
    )

    # --------------------------------------
    # Classification Filter
    # --------------------------------------

    selected_classification = st.selectbox(
        "Filter tweets by classification",
        options=[
            "All",
            "Complaint",
            "Compliment",
            "Request"
        ],
        index=0,
        key="classification_filter"
    )

    # --------------------------------------
    # Apply Multi-label Filter
    # --------------------------------------

    filtered_tweets = []

    for tweet in tweets:

        labels = tweet.classification or []

        if selected_classification == "All":

            filtered_tweets.append(tweet)

        else:

            selected_label = (
                selected_classification.lower()
            )

            if selected_label in labels:
                filtered_tweets.append(tweet)

    # --------------------------------------
    # Display Filter Results
    # --------------------------------------

    st.metric(
        label="Matching Tweets",
        value=len(filtered_tweets)
    )

    st.caption(
        f"Showing {len(filtered_tweets)} of "
        f"{len(tweets)} analyzed tweets."
    )

    # --------------------------------------
    # Prepare Results Table
    # --------------------------------------

    tweet_data = []

    for tweet in filtered_tweets:
        labels = tweet.classification or []

        tweet_data.append(
            {
                "Tweet ID": tweet.tweet_id,
                "Tweet Text": tweet.text,
                "Classification": ", ".join(labels),
                "Sentiment": tweet.sentiment
            }
        )

    tweet_df = pd.DataFrame(
        tweet_data,
        columns=[
            "Tweet ID",
            "Tweet Text",
            "Classification",
            "Sentiment"
        ]
    )

    # --------------------------------------
    # Display Results
    # --------------------------------------

    if tweet_df.empty:

        st.info(
            "No tweets found for the selected classification."
        )

    else:

        st.dataframe(
            tweet_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Tweet ID": st.column_config.TextColumn(
                    "Tweet ID",
                    alignment="center"
                ),
                "Tweet Text": st.column_config.TextColumn(
                    "Tweet Text",
                    width="large"
                ),
                "Classification": st.column_config.TextColumn(
                    "Classification",
                    alignment="center"
                ),
                "Sentiment": st.column_config.TextColumn(
                    "Sentiment",
                    alignment="center"
                )
            }
        )

    st.caption(
        "Multi-label classification: a tweet may appear "
        "in multiple classification categories."
    )

    st.divider()


    # ======================================
    # Model & Prompt Comparison
    # ======================================

    st.header("🧪 Model & Prompt Comparison")

    data_folder = (
        Path(__file__).resolve().parent.parent
        / "data"
    )

    evaluation_files = {
        "gpt-4o-mini / v2": "evaluation_results_v2.csv",
        "gpt-4o-mini / v3": "evaluation_results_v3.csv",
        "gpt-4.1-mini / v2": "evaluation_results_41_v2.csv",
        "gpt-4.1-mini / v3": "evaluation_results_41_v3.csv",
        "gpt-5-mini / v2": "evaluation_results_5_v2.csv",
        "gpt-5-mini / v3": "evaluation_results_5_v3.csv",
    }

    evaluation_data = []

    for experiment, filename in evaluation_files.items():

        file_path = data_folder / filename

        df = pd.read_csv(
            file_path,
            dtype={"tweet_id": str}
        )


        # ==================================
        # Sentiment Accuracy
        # ==================================

        sentiment_accuracy = (
            (
                df["predicted_sentiment"]
                == df["airline_sentiment"]
            ).mean() * 100
        )


        # ==================================
        # Classification Accuracy
        # ==================================

        labeled_df = df[
            df["classification_ground_truth"]
            != "needs_review"
        ]

        classification_accuracy = (
            (
                labeled_df["predicted_classification"]
                == labeled_df["classification_ground_truth"]
            ).mean() * 100
        )


        # ==================================
        # Macro F1 for Sentiment
        # ==================================

        sentiment_f1 = f1_score(
            df["airline_sentiment"],
            df["predicted_sentiment"],
            average="macro",
            zero_division=0
        ) * 100


        # ==================================
        # Macro F1 for Classification
        # ==================================

        classification_f1 = f1_score(
            labeled_df[
                "classification_ground_truth"
            ],
            labeled_df[
                "predicted_classification"
            ],
            average="macro",
            zero_division=0
        ) * 100


        evaluation_data.append(
            {
                "Experiment": experiment,
                "Tweets": len(df),
                "Sentiment Accuracy (%)": round(
                    sentiment_accuracy, 2
                ),
                "Classification Accuracy (%)": round(
                    classification_accuracy, 2
                ),
                "Sentiment Macro F1 (%)": round(
                    sentiment_f1, 2
                ),
                "Classification Macro F1 (%)": round(
                    classification_f1, 2
                )
            }
        )


    # ======================================
    # Evaluation Results Table
    # ======================================

    st.subheader("Evaluation Experiments")

    evaluation_df = pd.DataFrame(
        evaluation_data
    )

    st.dataframe(
        evaluation_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            col: st.column_config.Column(
                alignment="center"
            )
            for col in evaluation_df.columns
        }
    )


    # ======================================
    # Accuracy Comparison Chart
    # ======================================

    st.subheader("Model Accuracy Comparison")

    accuracy_df = pd.DataFrame(
        evaluation_data
    )

    accuracy_chart = accuracy_df.set_index(
        "Experiment"
    )[
        [
            "Sentiment Accuracy (%)",
            "Classification Accuracy (%)"
        ]
    ]

    st.bar_chart(
        accuracy_chart
    )


    # ======================================
    # Macro F1 Comparison Chart
    # ======================================

    st.subheader("Model Macro F1 Comparison")

    f1_chart = accuracy_df.set_index(
        "Experiment"
    )[
        [
            "Sentiment Macro F1 (%)",
            "Classification Macro F1 (%)"
        ]
    ]

    st.bar_chart(
        f1_chart
    )

    st.caption(
        "Evaluation results are based on historical "
        "single-label experiments using Prompt v2 and v3. "
        "Classification results are preliminary and are "
        "not an evaluation of the new multi-label Prompt v4."
    )


    # ======================================
    # Latency & Token Usage Comparison
    # ======================================

    st.header("⏱️ Latency & Token Usage Comparison")

    latency_file = (
        data_folder
        / "evaluation_latency_results.csv"
    )

    latency_df = pd.read_csv(
        latency_file,
        dtype={"tweet_id": str}
    )


    # ======================================
    # Latency Summary
    # ======================================

    latency_summary = (
        latency_df
        .groupby(
            ["model", "prompt_version"]
        )
        .agg(
            Tweets=(
                "tweet_id",
                "count"
            ),
            Average_Latency_Seconds=(
                "latency_seconds",
                "mean"
            ),
            Total_Input_Tokens=(
                "input_tokens",
                "sum"
            ),
            Total_Output_Tokens=(
                "output_tokens",
                "sum"
            ),
            Total_Tokens=(
                "total_tokens",
                "sum"
            )
        )
        .reset_index()
    )

    latency_summary[
        "Average_Latency_Seconds"
    ] = (
        latency_summary[
            "Average_Latency_Seconds"
        ].round(3)
    )


    # ======================================
    # Latency & Token Results Table
    # ======================================

    st.subheader(
        "Latency & Token Usage Results"
    )

    st.dataframe(
        latency_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            col: st.column_config.Column(
                alignment="center"
            )
            for col in latency_summary.columns
        }
    )


    # ======================================
    # Average Latency Comparison
    # ======================================

    st.subheader(
        "Average Latency Comparison"
    )

    latency_chart = latency_summary.copy()

    latency_chart["Experiment"] = (
        latency_chart["model"]
        + " / "
        + latency_chart["prompt_version"]
    )

    latency_chart = latency_chart.set_index(
        "Experiment"
    )

    st.bar_chart(
        latency_chart[
            "Average_Latency_Seconds"
        ]
    )


    # ======================================
    # Token Usage Comparison
    # ======================================

    st.subheader(
        "Token Usage Comparison"
    )

    token_chart = latency_summary.copy()

    token_chart["Experiment"] = (
        token_chart["model"]
        + " / "
        + token_chart["prompt_version"]
    )

    token_chart = token_chart.set_index(
        "Experiment"
    )

    st.bar_chart(
        token_chart[
            [
                "Total_Input_Tokens",
                "Total_Output_Tokens"
            ]
        ]
    )


    # ======================================
    # Estimated API Cost Comparison
    # ======================================

    st.header(
        "💰 Estimated API Cost Comparison (20 Tweets)"
    )

    pricing = {
        "gpt-4o-mini": {
            "input": 0.15,
            "output": 0.60
        },
        "gpt-4.1-mini": {
            "input": 0.40,
            "output": 1.60
        },
        "gpt-5-mini": {
            "input": 0.25,
            "output": 2.00
        }
    }


    # ======================================
    # Calculate Estimated Cost
    # ======================================

    cost_df = latency_summary.copy()

    cost_df[
        "Estimated Cost (USD)"
    ] = cost_df.apply(
        lambda row: (
            row["Total_Input_Tokens"]
            * pricing[row["model"]]["input"]
            +
            row["Total_Output_Tokens"]
            * pricing[row["model"]]["output"]
        ) / 1_000_000,
        axis=1
    )

    cost_df["Experiment"] = (
        cost_df["model"]
        + " / "
        + cost_df["prompt_version"]
    )


    # ======================================
    # Estimated Cost Results Table
    # ======================================

    st.subheader(
        "Estimated Cost Results"
    )

    cost_display_df = cost_df[
        [
            "Experiment",
            "Tweets",
            "Total_Input_Tokens",
            "Total_Output_Tokens",
            "Estimated Cost (USD)"
        ]
    ]

    st.dataframe(
        cost_display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            **{
                col: st.column_config.Column(
                    alignment="center"
                )
                for col in cost_display_df.columns
                if col != "Estimated Cost (USD)"
            },
            "Estimated Cost (USD)": (
                st.column_config.NumberColumn(
                    "Estimated Cost (USD)",
                    format="$%.6f",
                    alignment="center"
                )
            )
        }
    )


    # ======================================
    # Estimated Cost Comparison Chart
    # ======================================

    st.subheader(
        "Estimated Cost Comparison"
    )

    st.bar_chart(
        cost_df.set_index(
            "Experiment"
        )[
            "Estimated Cost (USD)"
        ]
    )

    st.caption(
        "Estimated costs are based on 20 identical tweets per "
        "model/prompt combination. Costs use the pricing "
        "assumptions recorded in Branch 6. Cached-token "
        "discounts, taxes and additional fees are excluded. "
        "Actual API billing may differ."
    )


# ==========================================
# Error Handling
# ==========================================

except Exception as e:

    st.error(
        "An error occurred while loading the dashboard."
    )

    st.exception(e)