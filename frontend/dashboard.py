import streamlit as st
import pandas as pd
import altair as alt
import requests
import re

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

def navigate_to(page):
    """Navigate using the same session state as the sidebar radio."""
    st.session_state["nav_page"] = page


# A normal browser navigation reloads the entire page and scrolls to its top.
# Unlike st.rerun(), this is a full browser refresh.
st.markdown(
    '<a class="airline-main-home" href="./" target="_self" '
    'title="Refresh Dashboard">✈️ Airline Feedback Dashboard</a>',
    unsafe_allow_html=True,
)

# ==========================================
# Custom Sidebar Styling
# ==========================================

st.markdown(
    """
    <style>
    /* Main and sidebar home links trigger a real page reload. */
    a.airline-main-home {
        display: inline-block;
        color: inherit !important;
        text-decoration: none !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        line-height: 1.25 !important;
    }
    a.airline-main-home:hover { opacity: 0.78; }

    /* Sidebar width */
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #F7F7F4;
    }
    
    /* Sidebar width only when expanded */
    [data-testid="stSidebar"][aria-expanded="true"] {
        width: 290px !important;
        min-width: 290px !important;
        max-width: 290px !important;
    }

    /* Sidebar navigation text */
    [data-testid="stSidebar"] [role="radiogroup"] p {
        font-size: 16px !important;
        font-weight: 400 !important;
        line-height: 1.6 !important;
}

    /* Navigation spacing */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 12px 8px !important;
        margin-bottom: 8px !important;
    }

    [data-testid="stSidebar"] a.airline-sidebar-home {
        color: inherit !important;
        text-decoration: none !important;
        font-size: 25px !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] a.airline-sidebar-home:hover { opacity: 0.75; }

    /* Blue Semantic Search button only; other buttons remain unchanged. */
    .st-key-semantic_search_submit button {
        background-color: #2563eb !important;
        border: 1px solid #2563eb !important;
        color: #ffffff !important;
    }
    .st-key-semantic_search_submit button p {
        color: #ffffff !important;
    }
    .st-key-semantic_search_submit button:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        color: #ffffff !important;
    }

    /* Help page: make clickable titles look like headings, not buttons */
    .st-key-help_dashboard button,
    .st-key-help_import button,
    .st-key-help_processed button {
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin-top: 16px !important;
        color: inherit !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        text-align: left !important;
        min-height: 0 !important;
    }
    .st-key-help_dashboard button p,
    .st-key-help_import button p,
    .st-key-help_processed button p {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        line-height: 1.35 !important;
    }
    .st-key-help_dashboard button:hover,
    .st-key-help_import button:hover,
    .st-key-help_processed button:hover {
        color: #3169a8 !important;
        text-decoration: underline !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ==========================================
# Sidebar Navigation
# ==========================================

with st.sidebar:

    st.markdown(
        '<a class="airline-sidebar-home" href="./" target="_self" '
        'title="Refresh Dashboard">✈️ Airline Feedback</a>',
        unsafe_allow_html=True,
    )

    st.divider()

    selected_page = st.radio(
        "Navigation",
        options=[
            "Dashboard",
            "Import New Tweets",
            "Processed Tweets",
            "Help"
        ],
        label_visibility="collapsed",
        key="nav_page",
    )


# Shared minimal, colorful theme for all four pages. No functional changes.
st.markdown(
    """
    <style>
    /* Shared page background without overwhelming the content. */
    [data-testid="stMain"] {
        background: linear-gradient(145deg, #f5f9ff 0%, #fffaf5 46%, #f6fbf9 100%);
    }
    /* A soft accent under each section title; chart colors remain unchanged. */
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3 {
        border-left: 4px solid #71a8c8;
        padding: 9px 14px;
        border-radius: 8px;
        background: linear-gradient(90deg, #e7f3ff 0%, #f5f3ff 46%, rgba(255,255,255,0) 100%);
    }
    /* Distinct but restrained chart panels, rather than floating on white. */
    [data-testid="stMain"] [data-testid="stVegaLiteChart"] {
        background: #ffffff;
        border: 1px solid #e8eaf4;
        border-radius: 14px;
        padding: 12px;
    }
    /* Make KPI pastel colors visible even when Streamlit wraps the border. */
    [class*="st-key-kpi_"] {
        border-radius: 14px;
    }
    .st-key-kpi_analyzed { background: #e7f1ff; }
    .st-key-kpi_complaints { background: #ffede8; }
    .st-key-kpi_compliments { background: #e4f6ee; }
    .st-key-kpi_requests { background: #f0e9ff; }
    /* Lightly tinted data table headers; no table content is changed. */
    [data-testid="stMain"] [data-testid="stDataFrame"] {
        border-radius: 10px;
        border: 1px solid #e6e7f0;
        overflow: hidden;
    }
    /* Consistent soft blue-green sidebar across all pages. */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eaf3ff 0%, #f1f8f5 100%) !important;
    }

    /* Quiet typography and softer, more consistent spacing. */
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3 {
        letter-spacing: -0.025em;
        font-weight: 650;
    }
    [data-testid="stMain"] [data-testid="stCaptionContainer"] {
        color: #737b86;
    }
    /* Subtle surfaces instead of heavy boxes. */
    [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e6e9ee;
        border-radius: 12px;
    }
    [data-testid="stMain"] [data-testid="stMetricValue"] {
        font-weight: 650;
        letter-spacing: -0.035em;
    }
    [data-testid="stMain"] hr {
        border-color: #e9edf1;
    }
    /* Soft colored KPI surfaces: expressive but still minimal. */
    .st-key-kpi_analyzed [data-testid="stVerticalBlockBorderWrapper"] {
        background: #eef5ff;
        border-color: #cfe0ff !important;
        border-left: 4px solid #5b8def !important;
    }
    .st-key-kpi_complaints [data-testid="stVerticalBlockBorderWrapper"] {
        background: #fff0ed;
        border-color: #f5d3cb !important;
        border-left: 4px solid #e77b70 !important;
    }
    .st-key-kpi_compliments [data-testid="stVerticalBlockBorderWrapper"] {
        background: #eaf8f2;
        border-color: #c5ead9 !important;
        border-left: 4px solid #38a58e !important;
    }
    .st-key-kpi_requests [data-testid="stVerticalBlockBorderWrapper"] {
        background: #f4effc;
        border-color: #e2d7f4 !important;
        border-left: 4px solid #9b83cf !important;
    }
    /* Preserve readable, strong numbers across pastel card colors. */
    .st-key-kpi_analyzed [data-testid="stMetricValue"],
    .st-key-kpi_complaints [data-testid="stMetricValue"],
    .st-key-kpi_compliments [data-testid="stMetricValue"],
    .st-key-kpi_requests [data-testid="stMetricValue"] {
        color: #233247;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# Help Page
# ==========================================

if selected_page == "Help":

    st.title("Help & User Guide")

    # Give the three clickable guide headings the same visual treatment
    # as the Future Improvements heading, without changing navigation.
    st.markdown(
        """
        <style>
        .st-key-help_dashboard,
        .st-key-help_import,
        .st-key-help_processed {
            border-left: 4px solid #71a8c8;
            border-radius: 8px;
            background: linear-gradient(90deg, #e7f3ff 0%, #f5f3ff 46%, rgba(255,255,255,0) 100%);
            padding: 9px 14px;
            margin-top: 16px;
        }
        .st-key-help_dashboard button,
        .st-key-help_import button,
        .st-key-help_processed button {
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .st-key-help_dashboard button p,
        .st-key-help_import button p,
        .st-key-help_processed button p {
            letter-spacing: -0.025em !important;
            font-weight: 650 !important;
            line-height: 1.35 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.button(
        "Dashboard",
        on_click=navigate_to,
        args=("Dashboard",),
        key="help_dashboard",
    )
    st.write(
        "View tweet statistics, classification and sentiment "
        "distributions, model comparisons, and estimated API costs."
    )

    st.button(
        "Import New Tweets",
        on_click=navigate_to,
        args=("Import New Tweets",),
        key="help_import",
    )
    st.write(
        "Upload a CSV file containing airline tweets. "
        "Preview the data, check duplicates, and start AI processing."
    )

    st.button(
        "Processed Tweets & Semantic Search",
        on_click=navigate_to,
        args=("Processed Tweets",),
        key="help_processed",
    )
    st.write(
        "Explore analyzed tweets, filter classifications, "
        "and search for relevant feedback by meaning."
    )

    st.subheader("Future Improvements")
    st.write(
        "Planned features include manual and automated "
        "ticket creation, customer support assignment, "
        "and workflow automation."
    )

    st.stop()
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
# Tweet Details (read-only)
# ==========================================

def get_tweet_details(tweet_id):
    """Read the original tweet and its latest analysis from PostgreSQL."""
    with SessionLocal() as db:
        comment = db.scalar(
            select(Comment).where(Comment.tweet_id == tweet_id)
        )
        if comment is None:
            return None

        analysis = db.scalar(
            select(Analysis)
            .where(Analysis.comment_id == comment.id)
            .order_by(Analysis.id.desc())
            .limit(1)
        )

        def field(obj, name):
            return getattr(obj, name, None) if obj is not None else None

        return {
            "comment_id": comment.id,
            "tweet_id": field(comment, "tweet_id"),
            "username": field(comment, "name"),
            "airline": field(comment, "airline"),
            "text": field(comment, "text"),
            "tweet_created": field(comment, "tweet_created"),
            "classification": field(analysis, "classification") or [],
            "sentiment": field(analysis, "sentiment"),
            "model": field(analysis, "model"),
            "prompt_version": field(analysis, "prompt_version"),
        }


def show_tweet_details(tweet_id):
    """Display details without altering the tweet or its classification."""
    details = get_tweet_details(tweet_id)
    if details is None:
        st.warning("This tweet is no longer available in PostgreSQL.")
        return

    st.subheader("Tweet Details")
    with st.container(border=True):
        st.caption("TWEET TEXT")
        st.write(clean_tweet_display(details["text"]) or "—")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Tweet ID**")
            st.write(details["tweet_id"] or "—")
            st.markdown("**Username**")
            st.write(details["username"] or "Not available")
            st.markdown("**Airline**")
            st.write(details["airline"] or "Not available")
            st.markdown("**Tweet date & time**")
            created = details["tweet_created"]
            st.write(str(created) if created else "Not available")
        with col2:
            st.markdown("**Classification**")
            st.write(", ".join(details["classification"]) or "Not available")
            st.markdown("**Sentiment**")
            st.write(details["sentiment"] or "Not available")
            st.markdown("**AI model / prompt**")
            st.write(
                f"{details['model'] or '—'} / {details['prompt_version'] or '—'}"
            )
            st.markdown("**Ticket status**")
            st.write("Not available — ticket management is planned")
        st.caption("Ticket number and assignee are not yet implemented.")


# ==========================================
# Remove confirmation popup (modal dialog)
# ==========================================

@st.dialog("Confirm Tweet Removal", width="small")
def confirm_tweet_removal(tweet_id):
    """Require explicit confirmation before permanently deleting a tweet."""
    details = get_tweet_details(tweet_id)
    if details is None:
        st.warning("This tweet is no longer available in PostgreSQL.")
        if st.button("Close", key="close_missing_remove"):
            st.session_state.pop("remove_tweet_id", None)
            st.rerun()
        return

    st.warning(
        "This permanently deletes the tweet, its embedding, "
        "and all related analyses. This action cannot be undone."
    )
    st.markdown(f"**Tweet ID:** `{details['tweet_id']}`")
    st.markdown(f"**Username:** {details['username'] or 'Not available'}")
    st.write(clean_tweet_display(details["text"]) or "—")

    cancel_col, confirm_col = st.columns(2)
    if cancel_col.button("Cancel", key="cancel_remove_popup", use_container_width=True):
        st.session_state.pop("remove_tweet_id", None)
        st.rerun()

    if confirm_col.button(
        "Confirm Delete", key="confirm_remove_popup",
        type="primary", use_container_width=True,
    ):
        try:
            response = requests.delete(
                f"{API_BASE_URL}/comments/{details['comment_id']}",
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            st.error("Could not delete the tweet. Please check FastAPI.")
            st.exception(exc)
        else:
            st.session_state.pop("remove_tweet_id", None)
            st.session_state.pop("view_tweet_id", None)
            st.session_state.pop("edit_tweet_id", None)
            st.session_state["remove_saved_feedback"] = True
            st.session_state.pop("semantic_search_results", None)
            st.rerun()


# ==========================================
# CSV Duplicate Detection
# ==========================================

def check_csv_duplicates(df):
    """
    Check tweet IDs against PostgreSQL
    and detect duplicates within the CSV.
    """

    tweet_ids = (
        df["tweet_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    with SessionLocal() as db:
        existing_ids = set(
            db.scalars(
                select(Comment.tweet_id).where(
                    Comment.tweet_id.in_(tweet_ids)
                )
            ).all()
        )

    seen_ids = set()
    statuses = []

    for tweet_id in tweet_ids:

        if not tweet_id:
            status = "Invalid"

        elif tweet_id in existing_ids:
            status = "Duplicate (Database)"

        elif tweet_id in seen_ids:
            status = "Duplicate (CSV)"

        else:
            status = "New"
            seen_ids.add(tweet_id)

        statuses.append(status)

    result = df.copy()
    result["Status"] = statuses

    return result
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


def clean_tweet_display(text):
    """Remove known synthetic-test wrappers for display; do not edit stored data."""
    cleaned = str(text or "")
    cleaned = re.sub(r"^\s*\[SYNTHETIC[ _-]*TEST[^]]*\]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*Example\s*#?\s*\d+\.?\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def format_tweet_preview(text, max_chars=20):
    """Show up to 20 characters of the cleaned tweet text in a table."""
    cleaned = " ".join(clean_tweet_display(text).split())
    return cleaned[:max_chars] + "..." if len(cleaned) > max_chars else cleaned


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
                "Tweet Text": format_tweet_preview(tweet.get("text", "")),
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
    all_successful = completed and failed == 0

    if all_successful:

        progress_bar.progress(1.0)

        status_text.success(
            "CSV processing completed successfully! "
            "All tweets were processed or identified as duplicates."
        )

    elif not completed:

        status_text.error(
            "CSV processing stopped before completion. "
            "Please check the processing errors."
        )

    else:

        status_text.warning(
            "CSV processing finished with errors. "
            "Some tweets could not be processed."
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
# Import New Tweets Page
# ==========================================

if selected_page == "Import New Tweets":

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

                preview_df = check_csv_duplicates(csv_df)


                def highlight_status(row):

                    if row["Status"].startswith("Duplicate"):
                        color = "background-color: #ffcccc"

                    elif row["Status"] == "New":
                        color = "background-color: #ccffcc"

                    else:
                        color = "background-color: #fff2cc"

                    return [color] * len(row)


                st.dataframe(
                    preview_df.style.apply(
                        highlight_status,
                        axis=1
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=300
                )

                new_count = (
                        preview_df["Status"] == "New"
                ).sum()

                duplicate_count = (
                    preview_df["Status"].str.startswith("Duplicate")
                ).sum()

                invalid_count = (
                        preview_df["Status"] == "Invalid"
                ).sum()

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Total Tweets", len(preview_df))
                col2.metric("New", int(new_count))
                col3.metric("Duplicates", int(duplicate_count))
                col4.metric("Invalid", int(invalid_count))



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

        if summary["Completed"] and summary["Failed"] == 0:

            st.success(
                "Batch processing completed successfully! "
                "All tweets were processed or identified as duplicates."
            )

        elif not summary["Completed"]:

            st.error(
                "Batch processing stopped before completion. "
                "Please check the errors below."
            )

        else:

            st.warning(
                "Batch processing finished with errors. "
                "Some tweets could not be processed."
            )

        if summary["Errors"]:

            st.subheader("Processing Errors")

            st.dataframe(
                pd.DataFrame(summary["Errors"]),
                use_container_width=True,
                hide_index=True
            )


    st.divider()



    # Do not render the other pages after the import page.
    st.stop()


# ==========================================
# Processed Tweets Page
# ==========================================

if selected_page == "Processed Tweets":
    # ==========================================
    # Semantic Search Interface
    # ==========================================

    st.header("🔎 Semantic Search")

    # Keep the search controls compact and aligned to the same width.
    search_form_col, _ = st.columns([1.5, 1.4])
    with search_form_col:
        with st.form("semantic_search_form"):

            search_query = st.text_input(
                "Enter a search query",
                placeholder="e.g. lost baggage"
            )

            # Dropdown of consecutive numbers, also accepts a typed value.
            search_limit_choice = st.selectbox(
                "Maximum number of results",
                options=list(range(1, 51)),
                index=4,
                accept_new_options=True,
                help="Choose 1–50 from the list or type a number and press Enter.",
            )

            search_button = st.form_submit_button(
                "🔍 Search",
                type="secondary",
                use_container_width=True,
                key="semantic_search_submit",
            )


    if search_button:

        # New selectbox values are strings; listed options are integers.
        chosen_value = str(search_limit_choice or "").strip()
        if not chosen_value.isdecimal() or not 1 <= int(chosen_value) <= 50:
            st.warning("Choose or enter a whole number between 1 and 50.")

        elif not search_query.strip():

            st.warning(
                "Please enter a search query."
            )

        elif len(search_query.strip()) < 2:

            st.warning(
                "Search query must contain at least "
                "2 characters."
            )

        else:
            search_limit = int(chosen_value)

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

    # Existing exact multi-label filters and tweet table
    try:
        tweets = get_latest_analyses()
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
                "Request",
                "Complaint + Request",
                "Complaint + Compliment",
                "Compliment + Request",
                "Complaint + Compliment + Request"
            ],
            index=0,
            key="classification_filter"
        )

        # --------------------------------------
        # Apply Multi-label Filter
        # --------------------------------------

        filtered_tweets = []

        for tweet in tweets:

            labels = set(tweet.classification or [])

            if selected_classification == "All":

                filtered_tweets.append(tweet)

            else:

                selected_labels = set(
                    selected_classification.lower().split(" + ")
                )

                if labels == selected_labels:
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
                    "Tweet Text": format_tweet_preview(tweet.text),
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
        # Display Results — one View action per tweet
        # --------------------------------------

        if st.session_state.pop("edit_saved_feedback", False):
            st.success("Classification and sentiment updated successfully.")
        if st.session_state.pop("remove_saved_feedback", False):
            st.success("Tweet and its analyses deleted successfully.")

        if tweet_df.empty:
            st.info("No tweets found for the selected classification.")
        else:
            # Show ten tweets per page; keep navigation below the rows.
            page_size = 10
            total_pages = max(1, (len(tweet_df) + page_size - 1) // page_size)
            page_key = "processed_tweets_page"
            current_page = st.session_state.get(page_key, 1)
            st.session_state[page_key] = max(1, min(current_page, total_pages))
            page = st.session_state[page_key]
            start = (page - 1) * page_size
            visible_tweets = tweet_df.iloc[start:start + page_size]

            header = st.columns([2.1, 2.6, 1.7, 1.0, 3.6])
            for col, title in zip(
                header,
                ["Tweet ID", "Tweet Text", "Classification", "Sentiment", "Action"]
            ):
                col.markdown(f"**{title}**")
            st.divider()

            for _, row in visible_tweets.iterrows():
                cols = st.columns([2.1, 2.6, 1.7, 1.0, 3.6], vertical_alignment="center")
                cols[0].write(row["Tweet ID"])
                cols[1].write(row["Tweet Text"])
                cols[2].write(row["Classification"])
                cols[3].write(row["Sentiment"])
                view_col, edit_col, remove_col, ticket_col = cols[4].columns([1, 1, 1.1, 2], gap="small")
                if view_col.button(
                    "View",
                    key=f"view_{row['Tweet ID']}",
                    use_container_width=True,
                ):
                    st.session_state["view_tweet_id"] = row["Tweet ID"]
                    st.session_state.pop("edit_tweet_id", None)
                    st.session_state.pop("remove_tweet_id", None)
                if edit_col.button(
                    "Edit",
                    key=f"edit_{row['Tweet ID']}",
                    use_container_width=True,
                ):
                    st.session_state["edit_tweet_id"] = row["Tweet ID"]
                    st.session_state.pop("view_tweet_id", None)
                    st.session_state.pop("remove_tweet_id", None)
                if remove_col.button(
                    "Delete",
                    key=f"remove_{row['Tweet ID']}",
                    use_container_width=True,
                ):
                    st.session_state["remove_tweet_id"] = row["Tweet ID"]
                    st.session_state.pop("view_tweet_id", None)
                    st.session_state.pop("edit_tweet_id", None)
                ticket_col.button(
                    "Create Ticket",
                    key=f"ticket_{row['Tweet ID']}",
                    disabled=True,
                    help="Coming soon — ticket management is not yet implemented.",
                    use_container_width=True,
                )

            # Compact pagination below the table (Previous / pages / Next).
            if total_pages <= 7:
                page_items = list(range(1, total_pages + 1))
            elif page <= 4:
                page_items = [1, 2, 3, 4, 5, "…", total_pages]
            elif page >= total_pages - 3:
                page_items = [1, "…", *range(total_pages - 4, total_pages + 1)]
            else:
                page_items = [1, "…", page - 1, page, page + 1, "…", total_pages]

            # Style only pagination buttons, not the View buttons or navigation.
            st.markdown(
                """
                <style>
                [class*="st-key-tweets_pager_"] button {
                    min-height: 32px !important;
                    height: 32px !important;
                    padding: 0 4px !important;
                    border: none !important;
                    border-radius: 6px !important;
                    background: transparent !important;
                    box-shadow: none !important;
                }
                [class*="st-key-tweets_pager_"] button p {
                    font-size: 15px !important;
                    font-weight: 400 !important;
                }
                [class*="st-key-tweets_pager_active_"] button p {
                    font-weight: 700 !important;
                    color: #1b365d !important;
                }
                [class*="st-key-tweets_pager_"] button:hover:not(:disabled) {
                    background: #f2f4f7 !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Center the pagination group without changing its buttons or logic.
            _, pager_center, _ = st.columns([1, 3, 1])
            controls = pager_center.columns([1] * (len(page_items) + 2))
            if controls[0].button(
                "❮", key="tweets_pager_previous", disabled=page == 1,
                use_container_width=True, help="Previous page",
            ):
                st.session_state[page_key] = page - 1
                st.rerun()

            for index, (column, item) in enumerate(zip(controls[1:-2], page_items)):
                if item == "…":
                    column.markdown("<div style='text-align:center; padding-top:4px'>…</div>", unsafe_allow_html=True)
                else:
                    if column.button(
                        str(item),
                        key=f"tweets_pager_active_{item}" if item == page else f"tweets_pager_page_{item}",
                        use_container_width=True,
                    ):
                        st.session_state[page_key] = item
                        st.rerun()

            if controls[-2].button(
                "❯", key="tweets_pager_next", disabled=page == total_pages,
                use_container_width=True, help="Next page",
            ):
                st.session_state[page_key] = page + 1
                st.rerun()

            st.caption(f"Page {page} of {total_pages} · {len(tweet_df)} matching tweets")

            # View, Edit, and Remove use a single selected tweet at a time.
            view_id = st.session_state.get("view_tweet_id")
            if view_id and view_id in set(tweet_df["Tweet ID"]):
                show_tweet_details(view_id)

            # Show the confirmation immediately in a centered popup.
            remove_id = st.session_state.get("remove_tweet_id")
            if remove_id and remove_id in set(tweet_df["Tweet ID"]):
                confirm_tweet_removal(remove_id)

            edit_id = st.session_state.get("edit_tweet_id")
            if edit_id and edit_id in set(tweet_df["Tweet ID"]):
                details = get_tweet_details(edit_id)
                if details is None:
                    st.warning("This tweet is no longer available in PostgreSQL.")
                else:
                    st.subheader("Edit Tweet Analysis")
                    with st.container(border=True):
                        st.caption(f"Tweet ID: {edit_id}")
                        st.write(clean_tweet_display(details["text"]) or "—")
                        with st.form(f"edit_analysis_form_{edit_id}"):
                            classification = st.multiselect(
                                "Classification",
                                options=["complaint", "compliment", "request"],
                                default=details["classification"],
                            )
                            sentiment_options = ["positive", "negative", "neutral"]
                            current_sentiment = details["sentiment"]
                            sentiment = st.selectbox(
                                "Sentiment",
                                options=sentiment_options,
                                index=(
                                    sentiment_options.index(current_sentiment)
                                    if current_sentiment in sentiment_options else 0
                                ),
                            )
                            save_changes = st.form_submit_button(
                                "Save Changes", type="primary"
                            )
                        if save_changes:
                            if not classification:
                                st.warning("Select at least one classification.")
                            else:
                                try:
                                    response = requests.patch(
                                        f"{API_BASE_URL}/comments/{details['comment_id']}/classification",
                                        json={
                                            "classification": classification,
                                            "sentiment": sentiment,
                                        },
                                        timeout=REQUEST_TIMEOUT,
                                    )
                                    response.raise_for_status()
                                except requests.exceptions.RequestException as exc:
                                    st.error("Could not save changes. Please check FastAPI.")
                                    st.exception(exc)
                                else:
                                    st.session_state.pop("edit_tweet_id", None)
                                    st.session_state["edit_saved_feedback"] = True
                                    st.rerun()

        st.caption(
            "Multi-label classification: a tweet may appear "
            "in multiple classification categories."
        )

        st.divider()
    except Exception as exc:
        st.error("Could not load processed tweets from PostgreSQL.")
        st.exception(exc)

    st.stop()


# ==========================================
# Main Dashboard
# ==========================================

if selected_page != "Dashboard":
    st.stop()


try:

    # ======================================
    # Overview — KPI Cards & Distributions
    # ======================================

    total_tweets = get_total_tweets()
    tweets = get_latest_analyses()

    classification_counts = Counter()
    sentiment_counts = Counter()
    combination_counts = Counter()
    label_order = ["complaint", "compliment", "request"]

    for tweet in tweets:
        labels = set(tweet.classification or [])
        classification_counts.update(labels)
        if tweet.sentiment:
            sentiment_counts.update([tweet.sentiment])
        ordered_labels = [label for label in label_order if label in labels]
        if ordered_labels:
            combination = " + ".join(label.title() for label in ordered_labels)
            combination_counts[combination] += 1

    st.markdown("### 📊 Feedback Overview")
    st.caption("Live overview of the latest classifications stored in PostgreSQL")
    kpi_items = [
        ("Analyzed Tweets", total_tweets, "✈️"),
        ("Complaints", classification_counts.get("complaint", 0), "📣"),
        ("Compliments", classification_counts.get("compliment", 0), "💚"),
        ("Requests", classification_counts.get("request", 0), "💬"),
    ]
    kpi_keys = ["kpi_analyzed", "kpi_complaints", "kpi_compliments", "kpi_requests"]
    for col, (label, value, icon), kpi_key in zip(st.columns(4), kpi_items, kpi_keys):
        with col:
            with st.container(border=True, key=kpi_key):
                st.caption(f"{icon} {label}")
                st.metric(label, value, label_visibility="collapsed")

    # Same underlying counts; multi-label tweets can contribute to several bars.
    classification_chart = pd.DataFrame({
        "Classification": ["Complaint", "Compliment", "Request"],
        "Count": [
            classification_counts.get("complaint", 0),
            classification_counts.get("compliment", 0),
            classification_counts.get("request", 0),
        ],
    })
    sentiment_chart = pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Count": [
            sentiment_counts.get("positive", 0),
            sentiment_counts.get("negative", 0),
            sentiment_counts.get("neutral", 0),
        ],
    })

    # Place both distributions side by side while keeping a tall chart area.
    classification_col, sentiment_col = st.columns([1.25, 1], gap="large")
    with classification_col:
        st.subheader("Classification Distribution")
        classification_plot = (
            alt.Chart(classification_chart)
            .mark_bar(cornerRadiusEnd=6, size=34)
            .encode(
                x=alt.X("Count:Q", title="Tweets", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("Classification:N", sort=["Complaint", "Compliment", "Request"], title=None),
                color=alt.Color(
                    "Classification:N",
                    scale=alt.Scale(
                        domain=["Complaint", "Compliment", "Request"],
                        range=["#EA786D", "#38A58E", "#B078A8"],
                    ),
                    legend=None,
                ),
                tooltip=["Classification:N", "Count:Q"],
            )
            .properties(height=360)
        )
        st.altair_chart(classification_plot, use_container_width=True)
        st.caption("A multi-label tweet can be counted in more than one category.")

    with sentiment_col:
        st.subheader("Sentiment Distribution")
        if sentiment_chart["Count"].sum() == 0:
            st.info("No sentiment data available.")
        else:
            sentiment_plot = (
                alt.Chart(sentiment_chart)
                .mark_arc(innerRadius=70, outerRadius=125)
                .encode(
                    theta=alt.Theta("Count:Q", stack=True),
                    color=alt.Color(
                        "Sentiment:N",
                        scale=alt.Scale(
                            domain=["Positive", "Negative", "Neutral"],
                            range=["#38A58E", "#EA786D", "#E6B653"],
                        ),
                        legend=alt.Legend(orient="bottom", title=None),
                    ),
                    tooltip=["Sentiment:N", "Count:Q"],
                )
                .properties(height=360)
                .configure_view(stroke=None)
            )
            st.altair_chart(sentiment_plot, use_container_width=True)
    with st.container(border=True):
        sentiment_cols = st.columns(3)
        for col, label in zip(sentiment_cols, ["positive", "negative", "neutral"]):
            with col:
                st.metric(label.title(), sentiment_counts.get(label, 0))

    st.divider()
    st.subheader("Multi-label Combination Distribution")
    combination_df = pd.DataFrame([
        {"Classification Combination": combination, "Tweets": count}
        for combination, count in combination_counts.items()
    ])
    if combination_df.empty:
        st.info("No classification data available.")
    else:
        combination_df = combination_df.sort_values("Tweets", ascending=False)
        combo_plot = (
            alt.Chart(combination_df)
            .mark_bar(cornerRadiusEnd=5, color="#D58A43")
            .encode(
                x=alt.X("Tweets:Q", title="Tweets", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("Classification Combination:N", sort="-x", title=None),
                tooltip=["Classification Combination:N", "Tweets:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(combo_plot, use_container_width=True)
        # Let the table size itself to actual rows; avoid blank filler rows.
        st.dataframe(combination_df, use_container_width=True, hide_index=True)
    st.caption("Each tweet is counted once in its exact classification combination.")
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

    accuracy_long = accuracy_chart.reset_index().melt(
        id_vars="Experiment", var_name="Metric", value_name="Score"
    )
    accuracy_plot = (
        alt.Chart(accuracy_long).mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Experiment:N", title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("Score:Q", title="Accuracy (%)", scale=alt.Scale(domain=[0, 100])),
            xOffset="Metric:N",
            color=alt.Color("Metric:N", scale=alt.Scale(range=["#DE9A46", "#38A58E"]), title=None),
            tooltip=["Experiment:N", "Metric:N", alt.Tooltip("Score:Q", format=".2f")],
        ).properties(height=300)
    )
    st.altair_chart(accuracy_plot, use_container_width=True)


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

    f1_long = f1_chart.reset_index().melt(
        id_vars="Experiment", var_name="Metric", value_name="Score"
    )
    f1_plot = (
        alt.Chart(f1_long).mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Experiment:N", title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("Score:Q", title="Macro F1 (%)", scale=alt.Scale(domain=[0, 100])),
            xOffset="Metric:N",
            color=alt.Color("Metric:N", scale=alt.Scale(range=["#9B83CF", "#50A9B3"]), title=None),
            tooltip=["Experiment:N", "Metric:N", alt.Tooltip("Score:Q", format=".2f")],
        ).properties(height=300)
    )
    st.altair_chart(f1_plot, use_container_width=True)

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

    latency_plot = (
        alt.Chart(latency_chart.reset_index()).mark_bar(color="#50A9B3", cornerRadiusEnd=4)
        .encode(
            x=alt.X("Experiment:N", title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("Average_Latency_Seconds:Q", title="Seconds"),
            tooltip=["Experiment:N", alt.Tooltip("Average_Latency_Seconds:Q", format=".3f")],
        ).properties(height=280)
    )
    st.altair_chart(latency_plot, use_container_width=True)


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

    token_long = token_chart.reset_index()[
        ["Experiment", "Total_Input_Tokens", "Total_Output_Tokens"]
    ].melt(id_vars="Experiment", var_name="Token Type", value_name="Tokens")
    token_plot = (
        alt.Chart(token_long).mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Experiment:N", title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("Tokens:Q", title="Tokens"),
            xOffset="Token Type:N",
            color=alt.Color("Token Type:N", scale=alt.Scale(range=["#B078A8", "#E6B653"]), title=None),
            tooltip=["Experiment:N", "Token Type:N", "Tokens:Q"],
        ).properties(height=280)
    )
    st.altair_chart(token_plot, use_container_width=True)


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

    cost_plot = (
        alt.Chart(cost_df).mark_bar(color="#5B9A86", cornerRadiusEnd=4)
        .encode(
            x=alt.X("Experiment:N", title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("Estimated Cost (USD):Q", title="Estimated cost (USD)"),
            tooltip=["Experiment:N", alt.Tooltip("Estimated Cost (USD):Q", format="$.6f")],
        ).properties(height=290)
    )
    st.altair_chart(cost_plot, use_container_width=True)

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