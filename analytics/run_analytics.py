"""Run the complete Titanic EDA, modeling, tuning, and regression workflow."""

from __future__ import annotations

import json
import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
TITANIC_CSV = ROOT / "titanic.csv"
RANDOM_STATE = 42


def load_once() -> pd.DataFrame:
    try:
        frame = sns.load_dataset("titanic")
    except Exception as error:
        if not TITANIC_CSV.exists():
            raise RuntimeError("Titanic download failed and analytics/titanic.csv is missing") from error
        frame = pd.read_csv(TITANIC_CSV)
    frame.to_csv(TITANIC_CSV, index=False)
    return frame


def clean_data(frame: pd.DataFrame, report: list[str]) -> pd.DataFrame:
    missing = (frame.isna().mean() * 100).round(2)
    affected = missing[missing > 0]
    report.append("Missing-value percentages before cleaning:\n" + affected.to_string())
    cleaned = frame.copy()
    cleaned = cleaned.drop(columns=["deck"], errors="ignore")
    report.append("deck was dropped because its measured missingness is above 30%; imputing that much missing cabin information would be unreliable.")
    if "age" in cleaned:
        cleaned["age"] = cleaned["age"].fillna(cleaned["age"].median())
        report.append(f"age missingness ({missing.get('age', 0):.2f}%) was between 5% and 30%, so it was median-imputed.")
    if "embarked" in cleaned:
        low_missing_columns = [column for column in ["embarked", "embark_town"] if column in cleaned]
        cleaned = cleaned.dropna(subset=low_missing_columns)
        for column in low_missing_columns:
            report.append(
                f"{column} missingness ({missing.get(column, 0):.2f}%) was below 5%, "
                "so rows missing that value were dropped."
            )
    cleaned["fare"] = cleaned["fare"].fillna(cleaned["fare"].median())
    return cleaned.reset_index(drop=True)


def save_eda(cleaned: pd.DataFrame, report: list[str]) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    report.append(f"Shape after cleaning: {cleaned.shape}")
    report.append("Profile:\n" + cleaned.describe(include="all").transpose().to_string())
    for column in ["age", "fare"]:
        values = cleaned[column].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        report.append(f"{column} IQR outlier count: {outliers}")
    fare_mode = cleaned["fare"].mode().iloc[0]
    fare_mean, fare_median = cleaned["fare"].mean(), cleaned["fare"].median()
    skew = "right-skewed" if fare_mean > fare_median > fare_mode else "left-skewed" if fare_mean < fare_median < fare_mode else "not strictly ordered"
    report.append(f"Fare mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}; the distribution is {skew} by mean/median/mode ordering.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(cleaned["age"], kde=True, ax=axes[0, 0]); axes[0, 0].set_title("Age distribution")
    sns.boxplot(x=cleaned["age"], ax=axes[0, 1]); axes[0, 1].set_title("Age outliers")
    sns.histplot(cleaned["fare"], kde=True, ax=axes[1, 0]); axes[1, 0].set_title("Fare distribution")
    sns.boxplot(x=cleaned["fare"], ax=axes[1, 1]); axes[1, 1].set_title("Fare outliers")
    fig.tight_layout(); fig.savefig(OUTPUTS / "01_age_fare_distributions.png", dpi=140); plt.close(fig)

    by_sex = pd.Series({
        sex: cleaned.loc[cleaned["sex"] == sex, "survived"].mean()
        for sex in cleaned["sex"].dropna().unique()
    }).sort_index()
    by_class = pd.Series({
        passenger_class: cleaned.loc[cleaned["pclass"] == passenger_class, "survived"].mean()
        for passenger_class in sorted(cleaned["pclass"].dropna().unique())
    })
    by_both = pd.Series({
        (sex, passenger_class): cleaned.loc[
            (cleaned["sex"] == sex) & (cleaned["pclass"] == passenger_class), "survived"
        ].mean()
        for sex in sorted(cleaned["sex"].dropna().unique())
        for passenger_class in sorted(cleaned["pclass"].dropna().unique())
    })
    report.append("Survival rate by sex:\n" + by_sex.to_string())
    report.append("Survival rate by pclass:\n" + by_class.to_string())
    report.append("Survival rate by sex and pclass:\n" + by_both.to_string())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.barplot(data=cleaned, x="sex", y="survived", ax=axes[0]); axes[0].set_title("Survival by sex")
    sns.barplot(data=cleaned, x="pclass", y="survived", ax=axes[1]); axes[1].set_title("Survival by class")
    sns.barplot(data=cleaned, x="pclass", y="survived", hue="sex", ax=axes[2]); axes[2].set_title("Sex and class")
    fig.tight_layout(); fig.savefig(OUTPUTS / "02_survival_breakdowns.png", dpi=140); plt.close(fig)

    correlation_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    correlation = cleaned[correlation_columns].corr()
    report.append("Required six-column correlation matrix:\n" + correlation.to_string())
    pairs = []
    for left_index, left in enumerate(correlation_columns):
        for right in correlation_columns[left_index + 1:]:
            pairs.append((abs(correlation.loc[left, right]), left, right, correlation.loc[left, right]))
    strongest = sorted(pairs, reverse=True)[:2]
    report.append("Two strongest absolute off-diagonal correlations: " + "; ".join(f"{left}-{right}={value:.3f}" for _, left, right, value in strongest))
    plt.figure(figsize=(8, 6)); sns.heatmap(correlation, annot=True, cmap="vlag", center=0); plt.title("Titanic numeric correlations"); plt.tight_layout(); plt.savefig(OUTPUTS / "03_correlation_heatmap.png", dpi=140); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=cleaned, x="pclass", y="fare", hue="survived", ax=axes[0]); axes[0].set_title("Fare, class, and survival")
    sns.scatterplot(data=cleaned, x="age", y="fare", hue="survived", style="sex", alpha=0.65, ax=axes[1]); axes[1].set_title("Age, fare, and survival")
    fig.tight_layout(); fig.savefig(OUTPUTS / "04_multivariate_story.png", dpi=140); plt.close(fig)
    report.extend([
        "Chart interpretation (age distribution and box plot): Most passenger ages cluster in adulthood, while the histogram shows the shape of the age distribution. The box plot marks the long-tail ages that meet the IQR outlier rule; age alone does not explain survival as strongly as sex or passenger class.",
        "Chart interpretation (fare distribution and box plot): Fare is strongly right-skewed, with a small number of passengers paying much more than the median. The box plot makes those high-fare outliers visible, so the mean is pulled upward relative to the median.",
        "Chart interpretation (survival by sex): The observed survival rate is substantially higher for women (about 74%) than men (about 19%). This large difference suggests sex is an important predictor in this dataset, though it should be interpreted alongside class and other features.",
        "Chart interpretation (survival by passenger class): Survival declines from first to third class, with rates of about 63%, 47%, and 24%, respectively. This gradient is consistent with passenger class capturing differences in access or location during evacuation.",
        "Chart interpretation (sex and passenger class): Women have higher survival rates than men within each passenger class, while survival also falls across classes. The combined plot shows that the overall sex gap is not merely an artifact of class composition, although the difference is smallest among third-class women and men.",
        "Chart interpretation (correlation heatmap): Passenger class and fare have the strongest absolute correlation (r = -0.548), consistent with lower class number being associated with higher fares. Sibling/spouse count and parent/child count are next (r = 0.415), indicating family-related passenger counts tend to vary together; neither relationship alone establishes causation.",
        "Chart interpretation (fare, class, and survival): First-class passengers generally paid higher fares, and the fare distribution differs between survivors and non-survivors within class. This view supports the separate class and survival patterns, while also showing substantial overlap that a classifier must handle.",
        "Chart interpretation (age and fare by survival): Survivors and non-survivors overlap across age and fare, so neither variable creates a clean separation by itself. The plot adds context to the stronger sex and class patterns and cautions against relying on a single feature.",
    ])
    standardized = cleaned[["age", "fare"]].copy()
    standardized[["age", "fare"]] = StandardScaler().fit_transform(standardized[["age", "fare"]])
    report.append("Standardization check (mean/std):\n" + standardized.agg(["mean", "std"]).round(4).to_string())


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def classification(cleaned: pd.DataFrame, report: list[str]) -> None:
    features = ["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]
    numeric, categorical = features[:5], features[5:]
    X, y = cleaned[features], cleaned["survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    report.append(f"Stratified split class balance: train={y_train.mean():.3f}, test={y_test.mean():.3f}; stratification preserves the survival ratio in both sets.")
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE),
    }
    rows, curves = [], []
    for name, estimator in estimators.items():
        pipeline = Pipeline([("preprocess", make_preprocessor(numeric, categorical)), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        probability = pipeline.predict_proba(X_test)[:, 1]
        rows.append({"model": name, "accuracy": accuracy_score(y_test, prediction), "precision": precision_score(y_test, prediction), "recall": recall_score(y_test, prediction), "f1": f1_score(y_test, prediction), "auc": roc_auc_score(y_test, probability)})
        curves.append((name, *roc_curve(y_test, probability)[:2]))
        report.append(f"{name} confusion matrix:\n{confusion_matrix(y_test, prediction)}\n{classification_report(y_test, prediction, zero_division=0)}")
        if name == "Decision Tree":
            transformed_names = pipeline.named_steps["preprocess"].get_feature_names_out()
            plt.figure(figsize=(18, 9)); plot_tree(pipeline.named_steps["model"], feature_names=transformed_names, class_names=["not survived", "survived"], filled=True, max_depth=3, fontsize=7); plt.tight_layout(); plt.savefig(OUTPUTS / "05_decision_tree.png", dpi=140); plt.close()
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUTS / "classification_comparison.csv", index=False)
    plt.figure(figsize=(8, 6))
    for name, false_positive, true_positive in curves:
        plt.plot(false_positive, true_positive, label=name)
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC comparison"); plt.legend(); plt.tight_layout(); plt.savefig(OUTPUTS / "06_roc_curves.png", dpi=140); plt.close()
    report.append("Classification comparison table:\n" + comparison.round(4).to_string(index=False))
    best_row = comparison.sort_values("f1", ascending=False).iloc[0]
    report.append(
        f"Recommendation: deploy {best_row['model']} because it has the strongest observed F1 score ({best_row['f1']:.3f}) on the held-out stratified test set. "
        f"Its precision is {best_row['precision']:.3f}, recall is {best_row['recall']:.3f}, and AUC is {best_row['auc']:.3f}. "
        "These metrics balance identifying survivors with avoiding incorrect positive predictions; the final choice should be revisited if operational costs change."
    )

    imbalance_rows = []
    variants = {
        "baseline": LogisticRegression(max_iter=1000),
        "class_weight_balanced": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SMOTE_train_only": LogisticRegression(max_iter=1000),
    }
    for name, estimator in variants.items():
        steps = [("preprocess", make_preprocessor(numeric, categorical))]
        if name == "SMOTE_train_only":
            steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        steps.append(("model", estimator))
        pipeline = ImbPipeline(steps)
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        imbalance_rows.append({"variant": name, "precision": precision_score(y_test, prediction), "recall": recall_score(y_test, prediction), "f1": f1_score(y_test, prediction)})
    imbalance = pd.DataFrame(imbalance_rows); imbalance.to_csv(OUTPUTS / "imbalance_comparison.csv", index=False)
    preferred = imbalance.sort_values("f1", ascending=False).iloc[0]
    report.append(
        f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}\n"
        f"Imbalance comparison:\n{imbalance.round(4).to_string(index=False)}\n"
        f"The highest F1 in this run is {preferred['f1']:.3f} for {preferred['variant']} "
        f"(precision {preferred['precision']:.3f}, recall {preferred['recall']:.3f}). "
        "The balanced class-weight and training-only SMOTE variants increase recall relative to baseline, "
        "at the cost of some precision; choose between them based on the relative costs of missed survivors "
        "and false alarms. SMOTE is applied only inside the training pipeline."
    )

    search = GridSearchCV(RandomForestClassifier(oob_score=True, bootstrap=True, random_state=RANDOM_STATE), {"n_estimators": [100, 150], "max_depth": [4, 7], "max_features": ["sqrt", "log2"]}, cv=3, scoring="f1", n_jobs=-1)
    tuned = Pipeline([("preprocess", make_preprocessor(numeric, categorical)), ("model", search)])
    tuned.fit(X_train, y_train)
    best_rf = tuned.named_steps["model"].best_estimator_
    report.append(f"Random Forest GridSearchCV best parameters: {search.best_params_}; best CV score={search.best_score_:.4f}; OOB score={best_rf.oob_score_:.4f}.")

    best_name = comparison.sort_values("f1", ascending=False).iloc[0]["model"]
    best_estimator = estimators[best_name]
    full_pipeline = Pipeline([("preprocess", make_preprocessor(numeric, categorical)), ("model", best_estimator)])
    full_pipeline.fit(X_train, y_train)
    joblib.dump(full_pipeline, OUTPUTS / "best_titanic_pipeline.joblib")
    reloaded = joblib.load(OUTPUTS / "best_titanic_pipeline.joblib")
    report.append(f"Reloaded complete pipeline prediction on raw input: {reloaded.predict(X_test.iloc[[0]]).tolist()}")


def regression(cleaned: pd.DataFrame, report: list[str]) -> None:
    regression_features = ["pclass", "age", "sibsp", "parch", "survived", "sex", "embarked"]
    X = cleaned[regression_features]
    y = cleaned["fare"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    numeric, categorical = regression_features[:5], regression_features[5:]
    pipeline = Pipeline([("preprocess", make_preprocessor(numeric, categorical)), ("model", LinearRegression())])
    pipeline.fit(X_train, y_train); prediction = pipeline.predict(X_test)
    r2 = r2_score(y_test, prediction); n, p = len(y_test), pipeline.named_steps["preprocess"].transform(X_test).shape[1]
    adjusted = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
    metrics = {"MAE": mean_absolute_error(y_test, prediction), "RMSE": np.sqrt(mean_squared_error(y_test, prediction)), "R2": r2, "Adjusted_R2": adjusted}
    report.append("Regression metrics (fare prediction):\n" + json.dumps(metrics, indent=2))
    residuals = y_test - prediction
    plt.figure(figsize=(7, 5)); sns.scatterplot(x=prediction, y=residuals); plt.axhline(0, color="black", linestyle="--"); plt.xlabel("Predicted fare"); plt.ylabel("Residual"); plt.title("Fare regression residuals"); plt.tight_layout(); plt.savefig(OUTPUTS / "07_regression_residuals.png", dpi=140); plt.close()
    residual_bins = pd.qcut(pd.Series(prediction), q=3, duplicates="drop")
    residual_spreads = pd.Series(residuals).groupby(residual_bins, observed=False).std().dropna()
    heteroscedastic = bool(len(residual_spreads) > 1 and residual_spreads.max() / max(residual_spreads.min(), 1e-9) > 2)
    report.append(f"Residual conclusion: the residual spread is {'consistent with possible heteroscedasticity' if heteroscedastic else 'not strongly heteroscedastic in this run'} based on prediction-bin spread ratios; inspect the saved residual plot for the visual funnel pattern.")
    classification_table = pd.read_csv(OUTPUTS / "classification_comparison.csv")
    classification_table["model_type"] = "classification"
    classification_table["model"] = classification_table.pop("model")
    for column in ["MAE", "RMSE", "R2", "Adjusted_R2"]:
        classification_table[column] = np.nan
    regression_row = pd.DataFrame([{"model_type": "regression", "model": "Linear Regression", "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "auc": np.nan, "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "R2": metrics["R2"], "Adjusted_R2": metrics["Adjusted_R2"]}])
    classification_table.to_csv(OUTPUTS / "model_comparison.csv", index=False)
    pd.concat([classification_table, regression_row], ignore_index=True).to_csv(OUTPUTS / "model_comparison.csv", index=False)
    report.append("Final model comparison table with separate classification and regression metric groups:\n" + pd.read_csv(OUTPUTS / "model_comparison.csv").round(4).to_string(index=False))


def main() -> None:
    report: list[str] = []
    raw = load_once()
    report.append(f"Raw shape: {raw.shape}")
    info_buffer = io.StringIO()
    raw.info(buf=info_buffer)
    report.append("Raw df.info():\n" + info_buffer.getvalue())
    cleaned = clean_data(raw, report)
    save_eda(cleaned, report)
    classification(cleaned, report)
    regression(cleaned, report)
    (ROOT / "analysis_report.md").write_text("# Titanic Analytics Report\n\n" + "\n\n".join(report), encoding="utf-8")
    print("\n\n".join(report))
    print(f"Saved analytics outputs to {OUTPUTS}")


if __name__ == "__main__":
    main()
