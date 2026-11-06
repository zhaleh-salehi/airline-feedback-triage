import csv
from pathlib import Path


DATA_DIR = Path("data")

FILES = {
    "GPT-4o-mini / v2": "evaluation_results_v2.csv",
    "GPT-4o-mini / v3": "evaluation_results_v3.csv",
    "GPT-4.1-mini / v3": "evaluation_results_41_v3.csv",
    "GPT-5-mini / v3": "evaluation_results_5_v3.csv",
}

CLASSIFICATION_LABELS = [
    "complaint",
    "compliment",
    "request",
]

SENTIMENT_LABELS = [
    "positive",
    "negative",
    "neutral",
]


def calculate_metrics(true_labels, predicted_labels, labels):

    total = len(true_labels)

    correct = sum(
        true == predicted
        for true, predicted in zip(true_labels, predicted_labels)
    )

    accuracy = correct / total if total else 0

    f1_scores = []

    for label in labels:

        tp = sum(
            true == label and predicted == label
            for true, predicted in zip(true_labels, predicted_labels)
        )

        fp = sum(
            true != label and predicted == label
            for true, predicted in zip(true_labels, predicted_labels)
        )

        fn = sum(
            true == label and predicted != label
            for true, predicted in zip(true_labels, predicted_labels)
        )

        precision = tp / (tp + fp) if (tp + fp) else 0

        recall = tp / (tp + fn) if (tp + fn) else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0
        )

        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(labels)

    return accuracy, macro_f1, correct, total


def print_confusion_matrix(true_labels, predicted_labels, labels):

    print("\nConfusion Matrix")
    print("Rows = Actual | Columns = Predicted")

    print(f"{'Actual':<15}", end="")

    for label in labels:
        print(f"{label:<15}", end="")

    print()

    for actual in labels:

        print(f"{actual:<15}", end="")

        for predicted in labels:

            count = sum(
                true == actual and pred == predicted
                for true, pred in zip(true_labels, predicted_labels)
            )

            print(f"{count:<15}", end="")

        print()


def evaluate_file(name, filename):

    file_path = DATA_DIR / filename

    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        rows = list(csv.DictReader(file))

    # Only use successful predictions
    rows = [
        row for row in rows
        if row["status"] == "success"
    ]

    # Sentiment evaluation
    sentiment_true = [
        row["airline_sentiment"]
        for row in rows
    ]

    sentiment_pred = [
        row["predicted_sentiment"]
        for row in rows
    ]

    # Classification evaluation
    classification_rows = [
        row for row in rows
        if row["classification_ground_truth"]
        in CLASSIFICATION_LABELS
    ]

    classification_true = [
        row["classification_ground_truth"]
        for row in classification_rows
    ]

    classification_pred = [
        row["predicted_classification"]
        for row in classification_rows
    ]

    # Calculate metrics
    sentiment_metrics = calculate_metrics(
        sentiment_true,
        sentiment_pred,
        SENTIMENT_LABELS
    )

    classification_metrics = calculate_metrics(
        classification_true,
        classification_pred,
        CLASSIFICATION_LABELS
    )

    # Token usage
    tokens = [
        int(row["tokens_used"])
        for row in rows
        if row["tokens_used"]
    ]

    total_tokens = sum(tokens)

    average_tokens = (
        total_tokens / len(tokens)
        if tokens else 0
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"\nSuccessful Tweets: {len(rows)}")

    print("\nSENTIMENT")
    print(f"Accuracy: {sentiment_metrics[0]:.2%}")
    print(f"Macro F1: {sentiment_metrics[1]:.4f}")
    print(
        f"Correct: {sentiment_metrics[2]}/{sentiment_metrics[3]}"
    )

    print_confusion_matrix(
        sentiment_true,
        sentiment_pred,
        SENTIMENT_LABELS
    )

    print("\nCLASSIFICATION")
    print(f"Accuracy: {classification_metrics[0]:.2%}")
    print(f"Macro F1: {classification_metrics[1]:.4f}")
    print(
        f"Correct: {classification_metrics[2]}/{classification_metrics[3]}"
    )

    print_confusion_matrix(
        classification_true,
        classification_pred,
        CLASSIFICATION_LABELS
    )

    print("\nTOKEN USAGE")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Average Tokens: {average_tokens:.2f}")


def main():

    for name, filename in FILES.items():
        evaluate_file(name, filename)


if __name__ == "__main__":
    main()