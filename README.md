# Zepto Data & AI Platform

This single repository contains the three rubric-aligned capstone modules: a normalized scraping/data pipeline, a Titanic analytics and modeling pipeline, and a grounded Zepto policy support assistant.

## Setup

The project uses one consolidated `requirements.txt` for all modules.

```powershell
cd C:\Users\91831\zepto_capstone
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Module 1: data pipeline

`data_pipeline/pipeline.py` scrapes Travel, Mystery, and Historical Fiction from books.toscrape.com, cleans fields, applies the fixed `1 GBP = 105.50 INR` rate, creates the `categories` and `books` PK/FK schema, runs six SQL demonstrations, and compares the JOIN through `pd.read_sql` and `pd.merge`.

```powershell
python data_pipeline\pipeline.py
```

Outputs: `data_pipeline/books.db` and `data_pipeline/query_results.txt`. The script requires at least 60 rows across three categories before loading.

## Module 2: analytics

`analytics/run_analytics.py` loads Seaborn's Titanic dataset once, saves `analytics/titanic.csv` as the offline fallback, cleans it using the stated missingness thresholds, writes the required EDA charts/report, trains three classifiers with train-only preprocessing, compares imbalance strategies, runs Random Forest GridSearchCV with OOB scoring, performs fare regression, and saves/reloads the complete best pipeline.

```powershell
python analytics\run_analytics.py
```

Outputs are written under `analytics/outputs`; the written interpretations are in `analytics/analysis_report.md`.

## Module 3: support assistant

`support_assistant/retrieval.py` loads the eight exact policy documents, embeds them with `all-MiniLM-L6-v2`, and stores them in ChromaDB. `support_assistant/main.py` provides the three-node LangGraph and FastAPI `POST /ask` endpoint. Mock mode is the default and needs no LLM API key.

```powershell
python -m support_assistant.retrieval
$env:MOCK_LLM = "1"
uvicorn support_assistant.main:app --reload
```

Test with:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" -Body '{"query":"How long does delivery take?"}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" -Body '{"query":"What is the capital of France?"}'
```

The first response is grounded with `doc_01_delivery_policy`; the second returns the fixed general-question response with an empty source list. Set `MOCK_LLM=0` and `GROQ_API_KEY` only for the optional real-LLM extension.

Build the required local container with:

```powershell
docker build -f support_assistant\Dockerfile -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
```

## Repository layout

```text
data_pipeline/       scraping, cleaning, SQLite schema, SQL evidence
analytics/            Titanic CSV, EDA/modeling script, charts, report
support_assistant/    eight docs, embeddings, Chroma, LangGraph, FastAPI, Dockerfile
requirements.txt      consolidated dependencies
```

Generated databases, model binaries, caches, and chart outputs are ignored by Git; every artifact is reproducible from the documented commands.
