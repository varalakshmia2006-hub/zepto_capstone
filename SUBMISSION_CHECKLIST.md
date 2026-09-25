# Submission Checklist

- [ ] Run `python data_pipeline/pipeline.py` and confirm at least 60 books across three categories.
- [ ] Run `python analytics/run_analytics.py` and confirm `analytics/titanic.csv`, charts, report, and model outputs.
- [ ] Build the Chroma index and run the FastAPI mock examples from `support_assistant/README.md`.
- [ ] Build the Docker image with `docker build -f support_assistant/Dockerfile -t zepto-support .`.
- [ ] Verify the public GitHub repository URL and submit the single repository link.

The feature branch used for this rebuild is intentionally merged back into `main` after multiple commits so the Git history demonstrates the required workflow.
