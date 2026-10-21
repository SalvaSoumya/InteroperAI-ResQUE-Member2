"""Train/evaluate a prototype disaster-priority classifier and anomaly detector.

Usage:
  pip install pandas scikit-learn joblib
  python train_disaster_models.py path/to/job-1791397398354_processed.csv

Important: the priority classifier is saved as a CANDIDATE only. Do not use it
for operational decisions unless its labels and evaluation results are validated.
"""
from pathlib import Path
import sys
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

REQUIRED = {
    "priority_level", "disaster_type", "affected_population", "reported_casualties",
    "injured_population", "displaced_population", "available_shelter_capacity",
    "current_shelter_occupancy", "estimated_loss", "required_resources",
    "available_resources", "infrastructure_damage", "incident_location",
    "incident_timestamp"
}
NUMERIC = [
    "affected_population", "reported_casualties", "injured_population",
    "displaced_population", "available_shelter_capacity",
    "current_shelter_occupancy", "estimated_loss", "latitude", "longitude",
]
CATEGORICAL = [
    "disaster_type", "incident_location", "required_resources",
    "available_resources", "infrastructure_damage",
]
TARGET = "priority_level"


def main(csv_path: str):
    source = Path(csv_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"CSV not found: {source}")
    out = source.parent / "member2_ml_artifacts"
    out.mkdir(exist_ok=True)

    df = pd.read_csv(source)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty")

    # Normalize strings and basic numeric fields.
    for c in CATEGORICAL + [TARGET]:
        df[c] = df[c].astype("string").str.strip()
        df[c] = df[c].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derive safe time features; avoid raw timestamp and post-response status.
    ts = pd.to_datetime(df["incident_timestamp"], errors="coerce")
    df["incident_month"] = ts.dt.month
    df["incident_hour"] = ts.dt.hour
    numeric_features = [c for c in NUMERIC if c in df.columns] + ["incident_month", "incident_hour"]
    categorical_features = [c for c in CATEGORICAL if c in df.columns]
    feature_cols = numeric_features + categorical_features

    labeled = df.dropna(subset=[TARGET]).copy()
    # If only one label exists, a supervised classifier cannot be evaluated.
    if labeled[TARGET].nunique() < 2:
        raise ValueError("priority_level has fewer than two classes; supervised classification is not possible.")

    X = labeled[feature_cols]
    y = labeled[TARGET].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", num_pipe, numeric_features),
        ("categorical", cat_pipe, categorical_features),
    ])
    classifier = Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42,
            min_samples_leaf=3, n_jobs=-1
        )),
    ])
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    base_pred = baseline.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    baseline_accuracy = accuracy_score(y_test, base_pred)
    balanced = balanced_accuracy_score(y_test, pred)

    report = {
        "dataset": str(source),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_cells_after_csv_load": int(df.isna().sum().sum()),
        "priority_class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts(dropna=False).items()},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features_used": feature_cols,
        "random_forest_accuracy": round(float(accuracy), 4),
        "majority_baseline_accuracy": round(float(baseline_accuracy), 4),
        "balanced_accuracy": round(float(balanced), 4),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "confusion_matrix_labels": sorted(y.unique().tolist()),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=sorted(y.unique().tolist())).tolist(),
        "recommendation": "Do not deploy the priority classifier if it is close to the majority baseline. Validate how priority labels were created and evaluate against trusted ground truth first."
    }
    # Save classifier only as a candidate, not a validated operational model.
    joblib.dump({"pipeline": classifier, "feature_columns": feature_cols, "target": TARGET,
                 "status": "candidate_not_validated"}, out / "priority_classifier_candidate.joblib")

    # Unsupervised anomaly detection on numeric incident/shelter/damage features.
    anomaly_cols = [c for c in NUMERIC if c in df.columns]
    anomaly_frame = df[anomaly_cols].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    anomaly_array = scaler.fit_transform(imputer.fit_transform(anomaly_frame))
    # Contamination is an assumed review fraction, not a known true anomaly rate.
    iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42, n_jobs=-1)
    labels = iso.fit_predict(anomaly_array)
    scores = iso.decision_function(anomaly_array)  # lower scores are more unusual
    df["anomaly_status"] = np.where(labels == -1, "Unusual", "Normal")
    df["anomaly_score"] = np.round(scores, 6)
    df["shelter_utilization_pct"] = np.where(
        df["available_shelter_capacity"] > 0,
        100 * df["current_shelter_occupancy"] / df["available_shelter_capacity"],
        np.nan,
    ).round(2)
    df.to_csv(out / "dataset_with_anomaly_scores.csv", index=False)
    joblib.dump({"imputer": imputer, "scaler": scaler, "model": iso,
                 "feature_columns": anomaly_cols,
                 "score_note": "Lower/negative decision_function scores indicate more unusual observations; scores are not probabilities."},
                out / "isolation_forest_model.joblib")

    report["anomaly_count"] = int((labels == -1).sum())
    report["anomaly_fraction"] = round(float((labels == -1).mean()), 4)
    report["anomaly_feature_columns"] = anomaly_cols
    report["shelter_utilization_summary_pct"] = df["shelter_utilization_pct"].describe().round(2).to_dict()
    report["data_quality_warnings"] = [
        "The supplied dataset has incident descriptions that may conflict with the disaster_type column; inspect before using text as a model feature.",
        "required_resources and available_resources appear to contain resource category names, not quantities. They cannot by themselves support a numeric shortage calculation.",
        "Isolation Forest anomaly labels are unsupervised flags for human review, not confirmed disaster anomalies.",
    ]
    with open(out / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    with open(out / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("MEMBER 2 — ML EXPERIMENT REPORT\n")
        f.write("=" * 36 + "\n")
        f.write(f"Dataset rows/columns: {len(df)} / {len(df.columns)}\n")
        f.write(f"Priority classes: {report['priority_class_counts']}\n")
        f.write(f"Random Forest accuracy: {accuracy:.2%}\n")
        f.write(f"Majority baseline accuracy: {baseline_accuracy:.2%}\n")
        f.write(f"Balanced accuracy: {balanced:.2%}\n")
        f.write(f"Isolation Forest unusual records: {report['anomaly_count']} ({report['anomaly_fraction']:.2%})\n\n")
        f.write("Classification report:\n")
        f.write(classification_report(y_test, pred, zero_division=0))
        f.write("\nCAVEATS\n")
        for warning in report["data_quality_warnings"]:
            f.write(f"- {warning}\n")
        f.write("- Do not claim predictive performance is reliable until labels are validated and metrics improve meaningfully over baseline.\n")
    print(f"Dataset: {len(df)} rows x {len(df.columns)} columns")
    print("Priority counts:", report["priority_class_counts"])
    print(f"Random Forest accuracy: {accuracy:.2%}")
    print(f"Majority baseline accuracy: {baseline_accuracy:.2%}")
    print(f"Balanced accuracy: {balanced:.2%}")
    print(f"Isolation Forest flagged: {report['anomaly_count']} ({report['anomaly_fraction']:.2%})")
    print(f"Artifacts saved to: {out}")
    print("NOTE: priority classifier is a candidate only; inspect evaluation_report.txt before any use.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_disaster_models.py path/to/processed.csv")
        raise SystemExit(2)
    main(sys.argv[1])
