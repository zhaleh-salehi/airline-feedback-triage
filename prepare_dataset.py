import csv
from pathlib import Path


# File paths
DATA_DIR = Path("data")

INPUT_FILE = DATA_DIR / "original_Tweets.csv"
OUTPUT_FILE = DATA_DIR / "Tweets_clean.csv"


# Columns we want to keep
COLUMNS = [
    "tweet_id",
    "airline_sentiment",
    "airline",
    "name",
    "text",
    "tweet_created",
]


def main():
    count = 0

    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)

        with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=COLUMNS)
            writer.writeheader()

            for row in reader:
                if row["airline"] == "United":
                    writer.writerow({
                        column: row[column]
                        for column in COLUMNS
                    })
                    count += 1

    print(f"Saved {count} United Airlines tweets")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()