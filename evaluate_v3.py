import csv
import os
from pathlib import Path

from analyzer_v3 import analyze_text, MODEL, PROMPT_VERSION

# File paths
INPUT_FILE = Path("data/evaluation_dataset_labeled_01_100_draft.csv")
OUTPUT_FILE = Path("data/evaluation_results_v3.csv")
# Evaluate 100 tweets
LIMIT = 100


def save_results(results, fieldnames):

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Save to a temporary file first
    temp_file = OUTPUT_FILE.with_name(OUTPUT_FILE.name + ".tmp")

    with open(
        temp_file,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    # Replace the previous file safely
    os.replace(temp_file, OUTPUT_FILE)


def main():

    # Read evaluation dataset
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)
        tweets = list(reader)
        input_fields = reader.fieldnames

    if not input_fields:
        raise ValueError("Evaluation dataset has no columns")

    print(f"Total evaluation tweets: {len(tweets)}")
    print(f"Model: {MODEL}")
    print(f"Prompt Version: {PROMPT_VERSION}")

    # Output columns
    extra_fields = [
        "predicted_classification",
        "predicted_sentiment",
        "tokens_used",
        "model",
        "prompt_version",
        "status",
    ]

    fieldnames = list(dict.fromkeys(input_fields + extra_fields))

    # Load previous results
    previous_results = {}

    if OUTPUT_FILE.exists():

        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                tweet_id = row["tweet_id"]

                if tweet_id in previous_results:
                    raise ValueError(
                        f"Duplicate tweet_id in results: {tweet_id}"
                    )

                previous_results[tweet_id] = row

        print(f"Previous results found: {len(previous_results)}")

    results = []
    successful = 0
    failed = 0
    skipped = 0

    # Process tweets one by one
    for index, tweet in enumerate(tweets[:LIMIT], start=1):

        tweet_id = tweet["tweet_id"]

        previous = previous_results.get(tweet_id)

        # Skip tweets already successfully analyzed
        if previous and previous["status"] == "success":

            if (
                previous.get("model") != MODEL
                or previous.get("prompt_version") != PROMPT_VERSION
            ):
                raise ValueError(
                    f"Model or prompt mismatch for tweet {tweet_id}"
                )

            print(f"Tweet {index}: Already processed. Skipping.")

            results.append(previous)
            skipped += 1

            continue

        print(f"\nProcessing tweet {index}/{min(LIMIT, len(tweets))}")

        try:

            analysis = analyze_text(tweet["text"])

            result = {
                **tweet,
                "predicted_classification": analysis["classification"],
                "predicted_sentiment": analysis["sentiment"],
                "tokens_used": analysis["tokens_used"],
                "model": MODEL,
                "prompt_version": PROMPT_VERSION,
                "status": "success",
            }

            successful += 1

            print("Classification:", analysis["classification"])
            print("Sentiment:", analysis["sentiment"])

        except Exception as error:

            print(f"FAILED: {error}")

            result = {
                **tweet,
                "predicted_classification": "",
                "predicted_sentiment": "",
                "tokens_used": "",
                "model": MODEL,
                "prompt_version": PROMPT_VERSION,
                "status": "failed",
            }

            failed += 1

        results.append(result)

        # Save after each processed tweet
        save_results(results, fieldnames)

    # Save final results
    save_results(results, fieldnames)

    print("\n--- Evaluation Summary ---")

    print("Total:", len(results))
    print("New Successful:", successful)
    print("Already Processed:", skipped)
    print("Failed:", failed)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()