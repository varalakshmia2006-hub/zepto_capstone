"""Run useful SQL queries against the scraped books database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "books.db"
OUTPUT_PATH = ROOT / "outputs" / "book_summary.csv"

QUERIES = {
    "Most expensive books": "SELECT title, price_gbp, rating FROM books ORDER BY price_gbp DESC LIMIT 10",
    "Books by rating": "SELECT rating, COUNT(*) AS book_count, ROUND(AVG(price_gbp), 2) AS average_price FROM books GROUP BY rating ORDER BY book_count DESC",
    "Price statistics": "SELECT COUNT(*) AS total_books, ROUND(MIN(price_gbp), 2) AS cheapest, ROUND(MAX(price_gbp), 2) AS most_expensive, ROUND(AVG(price_gbp), 2) AS average_price FROM books",
}


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit("Database not found. Run module1_data/scrape_and_store.py first.")
    with sqlite3.connect(DB_PATH) as connection:
        for label, query in QUERIES.items():
            print(f"\n--- {label} ---")
            print(pd.read_sql_query(query, connection).to_string(index=False))
        summary = pd.read_sql_query(
            "SELECT rating, COUNT(*) AS book_count, ROUND(AVG(price_gbp), 2) AS average_price FROM books GROUP BY rating ORDER BY rating",
            connection,
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved summary to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
