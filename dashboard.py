import streamlit as st
import pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path
from sqlalchemy import select, func

from database import SessionLocal
from models import Comment

st.set_page_config(
    page_title="Airline Feedback Dashboard",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Airline Feedback Dashboard")

st.write("Airline Feedback Triage System")

# Read the number of analyzed tweets from PostgreSQL
def get_total_tweets():
    with SessionLocal() as db:
        total = db.scalar(
            select(func.count(Comment.id))
        )
        return total


try:
    total_tweets = get_total_tweets()

    st.metric(
        label="Total Analyzed Tweets",
        value=total_tweets
    )

    st.success("Connected to PostgreSQL successfully!")
    # Count tweets by classification
    classification_counts = {}

    with SessionLocal() as db:
        results = db.execute(
            select(
                Comment.classification,
                func.count(Comment.id)
            ).group_by(Comment.classification)
        ).all()

        classification_counts = dict(results)

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
    # Classification Bar Chart
    st.subheader("Classification Chart")

    st.bar_chart(
        {
            "Complaint": classification_counts.get("complaint", 0),
            "Compliment": classification_counts.get("compliment", 0),
            "Request": classification_counts.get("request", 0)
        }
    )
    # Count tweets by sentiment
    with SessionLocal() as db:
        results = db.execute(
            select(
                Comment.sentiment,
                func.count(Comment.id)
            ).group_by(Comment.sentiment)
        ).all()

        sentiment_counts = dict(results)

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
    # Sentiment Bar Chart
    st.subheader("Sentiment Chart")

    st.bar_chart(
        {
            "Positive": sentiment_counts.get("positive", 0),
            "Negative": sentiment_counts.get("negative", 0),
            "Neutral": sentiment_counts.get("neutral", 0)
        }
    )
    # Display analyzed tweets
    st.subheader("Analyzed Tweets")

    with SessionLocal() as db:
        tweets = db.execute(
            select(
                Comment.tweet_id,
                Comment.text,
                Comment.classification,
                Comment.sentiment
            ).order_by(Comment.id.desc())
        ).all()

    tweet_data = [
        {
            "Tweet ID": tweet.tweet_id,
            "Tweet Text": tweet.text,
            "Classification": tweet.classification,
            "Sentiment": tweet.sentiment
        }
        for tweet in tweets
    ]

    tweet_df = pd.DataFrame(tweet_data)

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
    # Model & Prompt Comparison
    st.header("🧪 Model & Prompt Comparison")

    data_folder = Path(__file__).parent / "data"

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

        df = pd.read_csv(file_path, dtype={"tweet_id": str})

        # Sentiment Accuracy (100 tweets)
        sentiment_accuracy = (
                                     df["predicted_sentiment"]
                                     == df["airline_sentiment"]
                             ).mean() * 100

        # Classification Accuracy (84 labeled tweets)
        labeled_df = df[
            df["classification_ground_truth"] != "needs_review"
            ]

        classification_accuracy = (
                                          labeled_df["predicted_classification"]
                                          == labeled_df["classification_ground_truth"]
                                  ).mean() * 100

        # Macro F1 for Sentiment
        sentiment_f1 = f1_score(
            df["airline_sentiment"],
            df["predicted_sentiment"],
            average="macro",
            zero_division=0
        ) * 100

        # Macro F1 for Classification
        classification_f1 = f1_score(
            labeled_df["classification_ground_truth"],
            labeled_df["predicted_classification"],
            average="macro",
            zero_division=0
        ) * 100

        evaluation_data.append({
            "Experiment": experiment,
            "Tweets": len(df),
            "Sentiment Accuracy (%)": round(sentiment_accuracy, 2),
            "Classification Accuracy (%)": round(classification_accuracy, 2),
            "Sentiment Macro F1 (%)": round(sentiment_f1, 2),
            "Classification Macro F1 (%)": round(classification_f1, 2)
        })

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

    # Accuracy Comparison Chart
    st.subheader("Model Accuracy Comparison")

    accuracy_df = pd.DataFrame(evaluation_data)

    accuracy_chart = accuracy_df.set_index("Experiment")[
        [
            "Sentiment Accuracy (%)",
            "Classification Accuracy (%)"
        ]
    ]

    st.bar_chart(accuracy_chart)

    # Macro F1 Comparison Chart
    st.subheader("Model Macro F1 Comparison")

    f1_chart = accuracy_df.set_index("Experiment")[
        [
            "Sentiment Macro F1 (%)",
            "Classification Macro F1 (%)"
        ]
    ]

    st.bar_chart(f1_chart)

    # Latency & Token Usage Comparison
    st.header("⏱️ Latency & Token Usage Comparison")

    latency_file = data_folder / "evaluation_latency_results.csv"

    latency_df = pd.read_csv(
        latency_file,
        dtype={"tweet_id": str}
    )

    latency_summary = (
        latency_df
        .groupby(["model", "prompt_version"])
        .agg(
            Tweets=("tweet_id", "count"),
            Average_Latency_Seconds=("latency_seconds", "mean"),
            Total_Input_Tokens=("input_tokens", "sum"),
            Total_Output_Tokens=("output_tokens", "sum"),
            Total_Tokens=("total_tokens", "sum")
        )
        .reset_index()
    )

    latency_summary["Average_Latency_Seconds"] = (
        latency_summary["Average_Latency_Seconds"].round(3)
    )

    st.subheader("Latency & Token Usage Results")

    st.dataframe(
        latency_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            col: st.column_config.Column(alignment="center")
            for col in latency_summary.columns
        }
    )

    # Average Latency Comparison Chart
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

    # Token Usage Comparison Chart
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
            ["Total_Input_Tokens", "Total_Output_Tokens"]
        ]
    )

    # Estimated API Cost Comparison
    st.header("💰 Estimated API Cost Comparison (20 Tweets)")

    # Prices in USD per 1 million tokens
    # Same assumptions used in Branch 6
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

    cost_df = latency_summary.copy()

    # Calculate estimated cost
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

    # Display comparison table
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
                col: st.column_config.Column(alignment="center")
                for col in cost_display_df.columns
                if col != "Estimated Cost (USD)"
            },
            "Estimated Cost (USD)": st.column_config.NumberColumn(
                "Estimated Cost (USD)",
                format="$%.6f",
                alignment="center"
            )
        }
    )

    # Display cost comparison chart
    st.subheader("Estimated Cost Comparison")

    st.bar_chart(
        cost_df.set_index("Experiment")[
            "Estimated Cost (USD)"
        ]
    )

    st.caption(
        "Estimated costs are based on 20 identical tweets per "
        "model/prompt combination. Costs use the pricing assumptions "
        "recorded in Branch 6. Cached-token discounts, taxes and "
        "additional fees are excluded. Actual API billing may differ."
    )

except Exception as e:
    st.error("An error occurred while loading the dashboard.")
    st.exception(e)