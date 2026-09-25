"""Scrape, clean, store, and query a normalized books catalogue."""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data_pipeline" / "books.db"
QUERY_OUTPUT = ROOT / "data_pipeline" / "query_results.txt"
GBP_TO_INR = 105.50
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
CATEGORY_URLS = {
    "Travel": "https://books.toscrape.com/catalogue/category/books/travel_2/index.html",
    "Mystery": "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
    "Historical Fiction": "https://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html",
    "Science Fiction": "https://books.toscrape.com/catalogue/category/books/science-fiction_16/index.html",
}

SQL_QUERIES = {
    "select_where": "SELECT title, price_gbp, price_inr FROM books WHERE in_stock = 1 LIMIT 10;",
    "order_by_limit": "SELECT title, price_gbp FROM books ORDER BY price_inr DESC LIMIT 10;",
    "distinct_categories": "SELECT DISTINCT category_name FROM categories ORDER BY category_name;",
    "between_prices": "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 20 AND 40 ORDER BY price_gbp;",
    "in_ratings": "SELECT title, rating FROM books WHERE rating IN (4, 5) ORDER BY rating DESC, title LIMIT 10;",
    "join_top_rated": """SELECT c.category_name, b.title, b.rating, b.price_inr
        FROM books AS b JOIN categories AS c ON b.category_id = c.category_id
        WHERE b.rating >= 4 ORDER BY c.category_name, b.rating DESC, b.title LIMIT 15;""",
}


def parse_price(text: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group()) if match else None


def parse_book(card: Any, category_name: str) -> dict[str, Any] | None:
    title = card.select_one("h3 a")
    price = card.select_one(".price_color")
    rating_tag = card.select_one(".star-rating")
    availability_tag = card.select_one(".availability")
    if not all((title, price, rating_tag, availability_tag)):
        return None
    rating_text = next((value for value in rating_tag.get("class", []) if value in RATING_MAP), None)
    price_gbp = parse_price(price.get_text(" ", strip=True))
    if rating_text is None or price_gbp is None:
        return None
    availability = availability_tag.get_text(" ", strip=True)
    return {
        "title": title.get("title", title.get_text(" ", strip=True)).strip(),
        "price_gbp": price_gbp,
        "rating": RATING_MAP[rating_text],
        "in_stock": int("in stock" in availability.lower()),
        "availability_text": availability,
        "category_name": category_name,
    }


def scrape_catalogue() -> pd.DataFrame:
    session = requests.Session()
    session.headers["User-Agent"] = "ZeptoCapstone/2.0 educational scraper"
    rows: list[dict[str, Any]] = []
    for category_name, url in CATEGORY_URLS.items():
        response = session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        category_rows = [
            parsed
            for card in soup.select("article.product_pod")
            if (parsed := parse_book(card, category_name)) is not None
        ]
        rows.extend(category_rows)
        print(f"{category_name}: {len(category_rows)} books")
    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError("No books were parsed from the practice site")
    data["price_gbp"] = pd.to_numeric(data["price_gbp"], errors="coerce")
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
    data = data.dropna(subset=["price_gbp", "rating", "category_name"]).copy()
    data["price_gbp"] = data["price_gbp"].fillna(data["price_gbp"].median()).astype(float)
    data["rating"] = data["rating"].fillna(data["rating"].median()).round().clip(1, 5).astype(int)
    data["in_stock"] = data["in_stock"].astype(bool)
    data["price_inr"] = (data["price_gbp"] * GBP_TO_INR).round(2)
    return data


def create_database(data: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript("DROP TABLE IF EXISTS books; DROP TABLE IF EXISTS categories;")
        connection.execute("""CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE NOT NULL
        )""")
        connection.execute("""CREATE TABLE books (
            book_id INTEGER PRIMARY KEY, title TEXT NOT NULL, price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL, rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
            in_stock INTEGER NOT NULL CHECK(in_stock IN (0, 1)), availability_text TEXT NOT NULL,
            category_id INTEGER NOT NULL REFERENCES categories(category_id)
        )""")
        categories = data[["category_name"]].drop_duplicates().reset_index(drop=True)
        categories["category_id"] = categories.index + 1
        connection.executemany("INSERT INTO categories(category_id, category_name) VALUES (?, ?)", categories[["category_id", "category_name"]].itertuples(index=False, name=None))
        merged = data.merge(categories, on="category_name", how="left")
        connection.executemany(
            """INSERT INTO books(title, price_gbp, price_inr, rating, in_stock, availability_text, category_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            merged[["title", "price_gbp", "price_inr", "rating", "in_stock", "availability_text", "category_id"]].itertuples(index=False, name=None),
        )
        connection.commit()


def run_queries(db_path: Path = DB_PATH) -> None:
    outputs: list[str] = []
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        for name, query in SQL_QUERIES.items():
            frame = pd.read_sql_query(query, connection)
            outputs.append(f"\n--- {name} ---\n{query}\n{frame.to_string(index=False)}")
        books = pd.read_sql_query("SELECT * FROM books", connection)
        categories = pd.read_sql_query("SELECT * FROM categories", connection)
        sql_join = pd.read_sql_query(SQL_QUERIES["join_top_rated"], connection)
    pandas_join = books.merge(categories, on="category_id").query("rating >= 4").sort_values(["category_name", "rating", "title"], ascending=[True, False, True]).head(15)
    pandas_join = pandas_join[["category_name", "title", "rating", "price_inr"]].reset_index(drop=True)
    sql_join = sql_join.reset_index(drop=True)
    outputs.append(f"\n--- pandas.read_sql join ---\n{sql_join.to_string(index=False)}")
    outputs.append(f"\n--- pandas.merge equivalent ---\n{pandas_join.to_string(index=False)}")
    outputs.append(f"\nJOIN outputs equivalent: {sql_join.equals(pandas_join)}")
    QUERY_OUTPUT.write_text("\n".join(outputs), encoding="utf-8")
    print("\n".join(outputs))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-scrape", action="store_true", help="Reuse the existing database")
    args = parser.parse_args()
    if not args.skip_scrape:
        data = scrape_catalogue()
        if len(data) < 60 or data["category_name"].nunique() < 3:
            raise SystemExit("The final dataset must contain at least 60 books across 3 categories")
        create_database(data)
    if not DB_PATH.exists():
        raise SystemExit(f"Database not found: {DB_PATH}")
    run_queries()
    print(f"\nDatabase: {DB_PATH}\nQuery evidence: {QUERY_OUTPUT}")


if __name__ == "__main__":
    main()
