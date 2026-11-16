import streamlit as st
import pandas as pd

from collections import Counter
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

        # Find the latest analysis ID for each comment
        latest_analysis = (
            select(
                Analysis.comment_id,
                func.max(Analysis.id).label("latest_id")
            )
            .group_by(Analysis.comment_id)
            .subquery()
        )

        # Join comments with their latest analyses
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
# Main Dashboard
# ==========================================

try:

    # --------------------------------------
    # Total Tweets
    # --------------------------------------

    total_tweets = get_total_tweets()

    st.metric(
        label="Total Analyzed Tweets",
        value=total_tweets
    )

    st.success("Connected to PostgreSQL successfully!")


    # --------------------------------------
    # Load Latest Analyses
    # --------------------------------------

    tweets = get_latest_analyses()


    # ======================================
    # Classification Distribution
    # ======================================

    classification_counts = Counter()

    for tweet in tweets:

        labels = tweet.classification

        if labels:
            classification_counts.update(set(labels))


    st.subheader("Classification Distribution")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Complaints",
            classification_counts.get("complaint", 0)
        )

    with col2:

        st.metric(
            "Compliments",
            classification_counts.get("compliment", 0)
        )

    with col3:

        st.metric(
            "Requests",
            classification_counts.get("request", 0)
        )


    # Classification Chart

    st.subheader("Classification Chart")

    classification_chart = pd.DataFrame(
        {
            "Classification": [
                "Complaint",
                "Compliment",
                "Request"
            ],
            "Count": [
                classification_counts.get("complaint", 0),
                classification_counts.get("compliment", 0),
                classification_counts.get("request", 0)
            ]
        }
    )

    st.bar_chart(
        classification_chart.set_index("Classification")
    )

    st.caption(
        "Multi-label classification: one tweet may belong "
        "to more than one category."
    )


    # ======================================
    # Sentiment Distribution
    # ======================================

    sentiment_counts = Counter()

    for tweet in tweets:

        if tweet.sentiment:
            sentiment_counts.update([tweet.sentiment])


    st.subheader("Sentiment Distribution")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Positive",
            sentiment_counts.get("positive", 0)
        )

    with col2:

        st.metric(
            "Negative",
            sentiment_counts.get("negative", 0)
        )

    with col3:

        st.metric(
            "Neutral",
            sentiment_counts.get("neutral", 0)
        )


    # Sentiment Chart

    st.subheader("Sentiment Chart")

    sentiment_chart = pd.DataFrame(
        {
            "Sentiment": [
                "Positive",
                "Negative",
                "Neutral"
            ],
            "Count": [
                sentiment_counts.get("positive", 0),
                sentiment_counts.get("negative", 0),
                sentiment_counts.get("neutral", 0)
            ]
        }
    )

    st.bar_chart(
        sentiment_chart.set_index("Sentiment")
    )


    # ======================================
    # Analyzed Tweets
    # ======================================

    st.subheader("Analyzed Tweets")

    tweet_data = []

    for tweet in tweets:

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
                alignment="left"
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


    # ======================================
    # Model & Prompt Comparison
    # ======================================

    st.header("🧪 Model & Prompt Comparison")

    # Data folder is located in the project root
    data_folder = Path(__file__).resolve().parent.parent / "data"

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

        # ----------------------------------
        # Sentiment Accuracy
        # 100 tweets
        # ----------------------------------

        sentiment_accuracy = (
            (
                df["predicted_sentiment"]
                == df["airline_sentiment"]
            ).mean() * 100
        )


        # ----------------------------------
        # Classification Accuracy
        # 84 preliminary labeled tweets
        # ----------------------------------

        labeled_df = df[
            df["classification_ground_truth"] != "needs_review"
        ]

        classification_accuracy = (
            (
                labeled_df["predicted_classification"]
                == labeled_df["classification_ground_truth"]
            ).mean() * 100
        )


        # ----------------------------------
        # Macro F1 for Sentiment
        # ----------------------------------

        sentiment_f1 = f1_score(
            df["airline_sentiment"],
            df["predicted_sentiment"],
            average="macro",
            zero_division=0
        ) * 100


        # ----------------------------------
        # Macro F1 for Classification
        # ----------------------------------

        classification_f1 = f1_score(
            labeled_df["classification_ground_truth"],
            labeled_df["predicted_classification"],
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

    evaluation_df = pd.DataFrame(evaluation_data)

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

    accuracy_df = pd.DataFrame(evaluation_data)

    accuracy_chart = accuracy_df.set_index("Experiment")[
        [
            "Sentiment Accuracy (%)",
            "Classification Accuracy (%)"
        ]
    ]

    st.bar_chart(accuracy_chart)


    # ======================================
    # Macro F1 Comparison Chart
    # ======================================

    st.subheader("Model Macro F1 Comparison")

    f1_chart = accuracy_df.set_index("Experiment")[
        [
            "Sentiment Macro F1 (%)",
            "Classification Macro F1 (%)"
        ]
    ]

    st.bar_chart(f1_chart)

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
        data_folder / "evaluation_latency_results.csv"
    )

    latency_df = pd.read_csv(
        latency_file,
        dtype={"tweet_id": str}
    )


    # --------------------------------------
    # Latency Summary
    # --------------------------------------

    latency_summary = (
        latency_df
        .groupby(["model", "prompt_version"])
        .agg(
            Tweets=("tweet_id", "count"),
            Average_Latency_Seconds=(
                "latency_seconds", "mean"
            ),
            Total_Input_Tokens=(
                "input_tokens", "sum"
            ),
            Total_Output_Tokens=(
                "output_tokens", "sum"
            ),
            Total_Tokens=(
                "total_tokens", "sum"
            )
        )
        .reset_index()
    )

    latency_summary["Average_Latency_Seconds"] = (
        latency_summary[
            "Average_Latency_Seconds"
        ].round(3)
    )


    # ======================================
    # Latency & Token Results Table
    # ======================================

    st.subheader("Latency & Token Usage Results")

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

    st.subheader("Average Latency Comparison")

    latency_chart = latency_summary.copy()

    latency_chart["Experiment"] = (
        latency_chart["model"]
        + " / "
        + latency_chart["prompt_version"]
    )

    latency_chart = latency_chart.set_index("Experiment")

    st.bar_chart(
        latency_chart["Average_Latency_Seconds"]
    )


    # ======================================
    # Token Usage Comparison
    # ======================================

    st.subheader("Token Usage Comparison")

    token_chart = latency_summary.copy()

    token_chart["Experiment"] = (
        token_chart["model"]
        + " / "
        + token_chart["prompt_version"]
    )

    token_chart = token_chart.set_index("Experiment")

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

    # Prices in USD per 1 million tokens
    # Original pricing assumptions from Branch 6

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


    # --------------------------------------
    # Calculate Estimated Cost
    # --------------------------------------

    cost_df = latency_summary.copy()

    cost_df["Estimated Cost (USD)"] = cost_df.apply(
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

    st.subheader("Estimated Cost Results")

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

    st.subheader("Estimated Cost Comparison")

    st.bar_chart(
        cost_df.set_index("Experiment")[
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