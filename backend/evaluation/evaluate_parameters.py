import csv
import time
from pathlib import Path

from backend import analyzer_v3

# Experiment settings
MODEL = "gpt-4o-mini"

INPUT_FILE = Path(
    "../../data/evaluation_dataset_labeled_01_100_draft.csv"
)

OUTPUT_FILE = Path(
    "../../data/evaluation_parameter_results.csv"
)

LIMIT = 20


# Change only one parameter at a time
EXPERIMENTS = [
    {"name": "temperature_0", "temperature": 0.0, "max_tokens": 300},
    {"name": "temperature_05", "temperature": 0.5, "max_tokens": 300},
    {"name": "temperature_1", "temperature": 1.0, "max_tokens": 300},
    {"name": "tokens_100", "temperature": 0.0, "max_tokens": 100},
    {"name": "tokens_600", "temperature": 0.0, "max_tokens": 600},
]


FIELDNAMES = [
    "tweet_id",
    "experiment",
    "model",
    "prompt_version",
    "temperature",
    "max_completion_tokens",
    "classification",
    "sentiment",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "latency_seconds",
    "status",
    "error",
]


def save_results(results):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()
        writer.writerows(results.values())


def main():

    # Read five tweets
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8-sig"
    ) as file:

        tweets = list(csv.DictReader(file))[:LIMIT]

    if len(tweets) < LIMIT:
        raise ValueError("Not enough tweets in dataset")

    results = {}

    # Load previous results
    if OUTPUT_FILE.exists():

        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8-sig"
        ) as file:

            for row in csv.DictReader(file):

                key = (
                    row["tweet_id"],
                    row["experiment"]
                )

                results[key] = row

    # Run parameter experiments
    for experiment in EXPERIMENTS:

        print(f"\nExperiment: {experiment['name']}")

        for index, tweet in enumerate(tweets, start=1):

            key = (
                tweet["tweet_id"],
                experiment["name"]
            )

            if (
                key in results
                and results[key]["status"] == "success"
            ):
                print(f"Tweet {index}: Already processed")
                continue

            print(f"Processing tweet {index}/{LIMIT}")

            start = time.perf_counter()

            try:

                completion = (
                    analyzer_v3.client.chat.completions.parse(
                        model=MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": analyzer_v3.SYSTEM_PROMPT
                            },
                            {
                                "role": "user",
                                "content": tweet["text"]
                            }
                        ],
                        response_format=analyzer_v3.AnalysisResult,
                        temperature=experiment["temperature"],
                        max_completion_tokens=experiment["max_tokens"]
                    )
                )

                latency = time.perf_counter() - start

                parsed = completion.choices[0].message.parsed

                if parsed is None:
                    raise ValueError("Invalid AI response")

                usage = completion.usage

                if usage is None:
                    raise ValueError("Token usage unavailable")

                result = {
                    "tweet_id": tweet["tweet_id"],
                    "experiment": experiment["name"],
                    "model": MODEL,
                    "prompt_version": "v3",
                    "temperature": experiment["temperature"],
                    "max_completion_tokens": experiment["max_tokens"],
                    "classification": parsed.classification,
                    "sentiment": parsed.sentiment,
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "latency_seconds": round(latency, 3),
                    "status": "success",
                    "error": ""
                }

                print(
                    f"Result: {parsed.classification} / "
                    f"{parsed.sentiment}"
                )

            except Exception as error:

                latency = time.perf_counter() - start

                print(f"FAILED: {error}")

                result = {
                    "tweet_id": tweet["tweet_id"],
                    "experiment": experiment["name"],
                    "model": MODEL,
                    "prompt_version": "v3",
                    "temperature": experiment["temperature"],
                    "max_completion_tokens": experiment["max_tokens"],
                    "classification": "",
                    "sentiment": "",
                    "input_tokens": "",
                    "output_tokens": "",
                    "total_tokens": "",
                    "latency_seconds": round(latency, 3),
                    "status": "failed",
                    "error": str(error)
                }

            results[key] = result

            # Save results after every request
            save_results(results)

    print("\n--- Parameter Experiment Summary ---")

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