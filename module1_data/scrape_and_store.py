"""Scrape books.toscrape.com and store the results in SQLite."""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "books.db"


def scrape_books(max_pages: int) -> Iterable[dict]:
    session = requests.Session()
    session.headers.update({"User-Agent": "ZeptoCapstone/1.0 educational project"})
    for page_number in range(1, max_pages + 1):
        response = session.get(BASE_URL.format(page_number), timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        cards = soup.select("article.product_pod")
        if not cards:
            break
        for card in cards:
            title = card.h3.a.get("title", "").strip()
            price_text = card.select_one(".price_color").get_text(strip=True)
            price_match = re.search(r"\d+(?:\.\d+)?", price_text.replace(",", ""))
            if price_match is None:
                raise ValueError(f"Could not parse price: {price_text!r}")
            rating = card.select_one(".star-rating").get("class", ["", "Zero"])[1]
            availability = card.select_one(".availability").get_text(" ", strip=True)
            yield {
                "title": title,
                "price_gbp": float(price_match.group()),
                "rating": rating,
                "availability": availability,
                "source_page": page_number,
            }
        print(f"Scraped page {page_number}: {len(cards)} books")


def save_books(books: list[dict], db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE IF EXISTS books")
        connection.execute(
            """CREATE TABLE books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                rating TEXT NOT NULL,
                availability TEXT NOT NULL,
                source_page INTEGER NOT NULL,
                scraped_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        connection.executemany(
            """INSERT INTO books
               (title, price_gbp, rating, availability, source_page)
               VALUES (:title, :price_gbp, :rating, :availability, :source_page)""",
            books,
        )
        connection.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=5, help="Number of catalogue pages to scrape")
    args = parser.parse_args()
    if args.pages < 1:
        parser.error("--pages must be at least 1")
    try:
        books = list(scrape_books(args.pages))
    except requests.RequestException as error:
        raise SystemExit(f"Could not download the practice site: {error}") from error
    if not books:
        raise SystemExit("No books were found. The website layout may have changed.")
    save_books(books)
    print(f"Saved {len(books)} books to {DB_PATH}")


if __name__ == "__main__":
    main()
