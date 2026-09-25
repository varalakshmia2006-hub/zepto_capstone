# Data Pipeline

Run `python data_pipeline/pipeline.py` to scrape three public practice-site categories, clean the fields, compute `price_inr = price_gbp * 105.50`, create the normalized SQLite database, and print/save all required SQL evidence.

The schema has `categories(category_id, category_name)` and `books(book_id, ..., category_id)`, linked by a foreign key. Missing or malformed numeric values are median-imputed after coercion; rows missing required title/category/rating/price parsing are dropped because they cannot be identified or loaded safely. The script writes `books.db` and `query_results.txt` in this folder.
