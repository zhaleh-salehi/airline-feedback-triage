import csv
import time
from pathlib import Path

from backend import analyzer, analyzer_v3
from backend.experiments import analyzer_5_v2, analyzer_41_v2, analyzer_41, analyzer_5

INPUT_FILE = Path("../../data/evaluation_dataset_labeled_01_100_draft.csv")
OUTPUT_FILE = Path("../../data/evaluation_latency_results.csv")

LIMIT = 20

EXPERIMENTS = [
    analyzer,
    analyzer_v3,
    analyzer_41_v2,
    analyzer_41,
    analyzer_5_v2,
    analyzer_5,
]

FIELDNAMES = [
    "tweet_id",
    "model",
    "prompt_version",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "latency_seconds",
    "status",
]


def main():

    # Read the same 20 tweets for all experiments
    with open(INPUT_FILE, "r", encoding="utf-8-sig") as file:
        tweets = list(csv.DictReader(file))[:LIMIT]

    if len(tweets) < LIMIT:
        raise ValueError("Not enough tweets in evaluation dataset")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load results from earlier runs, if available
    results = {}

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                key = (
                    row["tweet_id"],
                    row["model"],
                    row["prompt_version"],
                )
                results[key] = row

    for experiment in EXPERIMENTS:

        model = experiment.MODEL
        prompt = experiment.PROMPT_VERSION

        print(f"\nExperiment: {model} / {prompt}")

        for index, tweet in enumerate(tweets, start=1):

            key = (tweet["tweet_id"], model, prompt)

            if key in results and results[key]["status"] == "success":
                print(f"Tweet {index}: Already processed")
                continue

            print(f"Processing tweet {index}/{LIMIT}")

            try:
                start = time.perf_counter()

                completion = experiment.client.chat.completions.parse(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": experiment.SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": tweet["text"],
                        },
                    ],
                    response_format=experiment.AnalysisResult,
                )

                latency = time.perf_counter() - start

                if completion.choices[0].message.parsed is None:
                    raise ValueError("Invalid AI response")

                usage = completion.usage

                if usage is None:
                    raise ValueError("Token usage unavailable")

                result = {
                    "tweet_id": tweet["tweet_id"],
                    "model": model,
                    "prompt_version": prompt,
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "latency_seconds": round(latency, 3),
                    "status": "success",
                }

            except Exception as error:

                print(f"FAILED: {error}")

                result = {
                    "tweet_id": tweet["tweet_id"],
                    "model": model,
                    "prompt_version": prompt,
                    "input_tokens": "",
                    "output_tokens": "",
                    "total_tokens": "",
                    "latency_seconds": "",
                    "status": "failed",
                }

            results[key] = result

            # Save progress after each request
            with open(
                OUTPUT_FILE,
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=FIELDNAMES,
                )

                writer.writeheader()
                writer.writerows(results.values())

    print("\n--- Latency Experiment Summary ---")

    successful = sum(
        row["status"] == "success"
        for row in results.values()
    )

    failed = sum(
        row["status"] == "failed"
        for row in results.values()
    )

    print("Total Records:", len(results))
    print("Successful:", successful)
    print("Failed:", failed)
    print("Results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()