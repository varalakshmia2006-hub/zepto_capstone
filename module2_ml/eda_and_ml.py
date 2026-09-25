"""Explore the scraped books and compare three classification models."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "books.db"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_book_model.joblib"


def load_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise SystemExit("Database not found. Run module1_data/scrape_and_store.py first.")
    with sqlite3.connect(DB_PATH) as connection:
        data = pd.read_sql_query("SELECT title, price_gbp, rating, availability, source_page FROM books", connection)
    data["rating_number"] = data["rating"].map({"Zero": 0, "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5})
    data["is_available"] = data["availability"].str.contains("In stock", case=False).astype(int)
    data["price_band"] = pd.qcut(data["price_gbp"], q=3, labels=["Budget", "Mid", "Premium"], duplicates="drop")
    return data


def main() -> None:
    data = load_data()
    print("\nEDA overview")
    print(data[["price_gbp", "rating_number", "is_available"]].describe().round(2).to_string())
    print("\nTarget distribution")
    print(data["price_band"].value_counts().to_string())

    features = ["rating_number", "is_available", "source_page"]
    X = data[features]
    y = data["price_band"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    preprocessor = ColumnTransformer(
        [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)],
        remainder="drop",
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    results = []
    fitted = {}
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results.append({"model": name, "accuracy": round(accuracy, 4)})
        fitted[name] = pipeline
        print(f"\n{name}: accuracy={accuracy:.4f}")
        print(classification_report(y_test, predictions, zero_division=0))

    results_frame = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_frame.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    best_name = results_frame.iloc[0]["model"]
    joblib.dump(fitted[best_name], MODEL_PATH)
    print(results_frame.to_string(index=False))
    print(f"Best model ({best_name}) saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
