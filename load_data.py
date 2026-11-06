import csv
import time
from pathlib import Path
from datetime import datetime

import requests


# Configuration
BASE_URL = "http://127.0.0.1:8000"
CSV_PATH = Path(__file__).resolve().parent / "data" / "Tweets_clean.csv"
LIMIT = 5
REQUEST_TIMEOUT = 60


def main():
    successful = 0
    duplicates = 0
    failed = 0
    processed = 0

    # Read CSV file
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if processed >= LIMIT:
                break

            processed += 1
            # Convert CSV datetime to ISO 8601 format
            raw_date = row["tweet_created"].strip()

            tweet_date = (
                datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S %z").isoformat()
                if raw_date
                else None
            )
            # Prepare one tweet
            tweet = {
                "tweet_id": row["tweet_id"].strip(),
                "airline_sentiment": row["airline_sentiment"] or None,
                "airline": row["airline"] or None,
                "name": row["name"] or None,
                "text": row["text"],
                "tweet_created": tweet_date,            }

            print(f"\nProcessing tweet {processed}/{LIMIT}")
            print(f"Tweet ID: {tweet['tweet_id']}")

            try:
                # Send one tweet to the existing FastAPI workflow
                response = requests.post(
                    f"{BASE_URL}/comments",
                    json=tweet,
                    timeout=REQUEST_TIMEOUT,
                )

                if response.status_code == 201:
                    successful += 1
                    print("SUCCESS: Tweet analyzed and saved")

                elif response.status_code == 409:
                    duplicates += 1
                    print("DUPLICATE: Tweet already exists")

                else:
                    failed += 1
                    print(f"FAILED: HTTP {response.status_code}")
                    print(response.text)

                    # Stop if the AI service or database is unavailable
                    if response.status_code in (500, 502):
                        print("Stopping batch due to server error")
                        break

            except requests.exceptions.RequestException as error:
                failed += 1
                print(f"CONNECTION ERROR: {error}")
                print("Stopping batch")
                break

            # Small pause between requests
            time.sleep(1)

    print("\n--- Batch Processing Summary ---")
    print(f"Processed: {processed}")
    print(f"Successful: {successful}")
    print(f"Duplicates: {duplicates}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()