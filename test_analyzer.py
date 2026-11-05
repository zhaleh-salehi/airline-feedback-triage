import csv

from analyzer import analyze_text


CSV_PATH = "data/Tweets.csv"
LIMIT = 5


with open(CSV_PATH, newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for i, row in enumerate(reader, start=1):
        if i > LIMIT:
            break

        tweet = row["text"]

        print(f"\n--- Tweet {i} ---")
        print(f"Airline: {row['airline']}")
        print(f"Text: {tweet}")
        print(f"Original Sentiment: {row['airline_sentiment']}")

        result = analyze_text(tweet)

        print(f"AI Classification: {result['classification']}")
        print(f"AI Sentiment: {result['sentiment']}")
        print(f"Tokens Used: {result['tokens_used']}")