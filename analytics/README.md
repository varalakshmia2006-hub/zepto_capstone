# Analytics Pipeline

Run `python analytics/run_analytics.py`. It calls `sns.load_dataset("titanic")` once, immediately commits the resulting offline fallback to `analytics/titanic.csv`, cleans that same frame, writes EDA charts and `analysis_report.md`, then continues with the leakage-safe classifier, imbalance, tuning, and regression workflows. All preprocessing is inside scikit-learn or imbalanced-learn pipelines fit on training data only.

The report states missingness percentages and threshold decisions, IQR outlier counts, fare skew, survival breakdowns, exact six-column correlations, chart interpretations, standardization checks, model metrics, GridSearchCV/OOB results, regression metrics, and the reload test for `outputs/best_titanic_pipeline.joblib`.
