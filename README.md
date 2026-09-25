# Zepto Capstone

A beginner-friendly three-module data and AI project:

1. Scrape a practice bookstore into SQLite and run SQL analysis.
2. Explore the data and compare three machine-learning classifiers.
3. Build a retrieval-augmented policy chatbot with an optional Groq API.

## 1. Setup

Open PowerShell in this project folder:

```powershell
cd C:\Users\91831\zepto_capstone
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 2. Module 1: scraping and SQL

This uses the safe practice site `books.toscrape.com`. An internet connection is needed for the first command.

```powershell
python module1_data\scrape_and_store.py --pages 5
python module1_data\sql_queries.py
```

The database is saved as `data\books.db`. A CSV summary is saved in `outputs\book_summary.csv`.

To scrape more or fewer pages, change `--pages`, for example `--pages 10`.

## 3. Module 2: EDA and machine learning

The raw bookstore data has no classification label. The script creates `price_band` from the lower, middle, and upper thirds of book prices, then compares Logistic Regression, Random Forest, and Gradient Boosting.

```powershell
python module2_ml\eda_and_ml.py
```

Results are saved to `outputs\model_comparison.csv`, and the best fitted pipeline is saved to `outputs\best_book_model.joblib`.

## 4. Module 3: RAG policy chatbot

The chatbot already includes three small policy documents in `module3_rag\docs`. It can answer using local retrieval without an API key:

```powershell
python module3_rag\rag_chatbot.py
```

Type a question, then type `quit` to exit. To skip the sample questions:

```powershell
python module3_rag\rag_chatbot.py --no-samples
```

### Optional Groq answers

1. Create an API key at [console.groq.com](https://console.groq.com/).
2. Set it for the current PowerShell session:

```powershell
$env:GROQ_API_KEY = "paste-your-key-here"
python module3_rag\rag_chatbot.py
```

The key is read from the environment and is never stored in the project. You can change the model with `GROQ_MODEL`; the default is `llama-3.1-8b-instant`.

## Project structure

```text
zepto_capstone/
|-- data/                    generated SQLite database
|-- outputs/                 generated CSV and model files
|-- module1_data/
|   |-- scrape_and_store.py
|   `-- sql_queries.py
|-- module2_ml/
|   `-- eda_and_ml.py
|-- module3_rag/
|   |-- docs/
|   |   |-- privacy_policy.txt
|   |   |-- returns_policy.txt
|   |   `-- shipping_policy.txt
|   `-- rag_chatbot.py
|-- requirements.txt
`-- README.md
```

## Common fixes

- `ModuleNotFoundError`: activate `.venv` and run `pip install -r requirements.txt` again.
- `Database not found`: run the Module 1 scraper before Module 1 SQL or Module 2.
- Scraping timeout: check the internet connection and rerun the scraper later.
- No Groq key: this is okay; the chatbot automatically uses local extractive answers.
