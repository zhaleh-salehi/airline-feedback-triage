import csv
from collections import defaultdict
from pathlib import Path

# Input and output prices in USD per 1 million tokens
# Standard API rates used for this experiment
PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
}

INPUT_FILE = Path("data/evaluation_latency_results.csv")


def main():

    totals = defaultdict(
        lambda: {"input": 0, "output": 0, "count": 0}
    )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row["status"] != "success":
                continue

            model = row["model"]
            prompt = row["prompt_version"]

            key = (model, prompt)

            totals[key]["input"] += int(row["input_tokens"])
            totals[key]["output"] += int(row["output_tokens"])
            totals[key]["count"] += 1

    print("\n--- Cost Comparison ---")

    for (model, prompt), usage in totals.items():

        prices = PRICES[model]

        input_cost = (
            usage["input"] * prices["input"] / 1_000_000
        )

        output_cost = (
            usage["output"] * prices["output"] / 1_000_000
        )

        total_cost = input_cost + output_cost

        print(f"\nModel: {model}")
        print(f"Prompt: {prompt}")
        print(f"Tweets: {usage['count']}")
        print(f"Input Tokens: {usage['input']}")
        print(f"Output Tokens: {usage['output']}")
        print(f"Estimated Cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()