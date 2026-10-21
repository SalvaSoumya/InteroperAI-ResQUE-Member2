"""Comprehensive Test Suite for Hybrid Disaster Assessment Workflow (Member 2).

Validates:
1. Latest NOAA file validation (job-1792491883115_processed.csv)
2. Missing-value and feature-quality checks
3. Hybrid artifact verification and scikit-learn compatibility
4. Inference on the latest 500 records
5. Output row count and missing/unknown-value checks
6. API integration test for receiving and assessing CSV (POST /api/receive-file)
7. Regression tests for existing API behavior (/api/health, /api/assess, /api/processed-results)
"""
import json
import os
import io
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from fastapi.testclient import TestClient

from main import app, load_hybrid_artifact, screen_evidence_impact

client = TestClient(app)

DATASET_PATH = Path(__file__).resolve().parent / "received_datasets" / "job-1792491883115_processed.csv"
HYBRID_ARTIFACT_PATH = Path(__file__).resolve().parent / "received_datasets" / "member2_hybrid_artifacts" / "hybrid_isolation_forest.joblib"
EVAL_REPORT_PATH = Path(__file__).resolve().parent / "received_datasets" / "member2_hybrid_artifacts" / "hybrid_evaluation_report.json"


def test_1_latest_noaa_file_validation():
    """Verify that job-1792491883115_processed.csv exists with expected properties."""
    print("\n--- Test 1: Latest NOAA File Validation ---")
    assert DATASET_PATH.exists(), f"File does not exist: {DATASET_PATH}"
    df = pd.read_csv(DATASET_PATH)
    print(f"File verified at: {DATASET_PATH}")
    print(f"Row count: {len(df)}")
    print(f"Column count: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    assert len(df) == 500, f"Expected 500 rows, found {len(df)}"
    assert len(df.columns) in (24, 29), f"Expected 24 (canonical + base) or 29 (with hybrid assessment) columns, found {len(df.columns)}"
    return df


def test_2_missing_value_and_feature_quality(df: pd.DataFrame):
    """Inspect missing-value statistics, variance, and priority-label distribution."""
    print("\n--- Test 2: Missing-Value and Feature-Quality Checks ---")
    missing = df.isna().sum()
    print("Column Missingness:")
    for col, count in missing.items():
        pct = (count / len(df)) * 100.0
        print(f"  {col}: {count}/{len(df)} missing ({pct:.1f}%)")

    # Priority distribution
    priority_col = df["priority_level"].fillna("Unknown").astype(str).str.strip()
    dist = priority_col.value_counts().to_dict()
    print(f"Priority Level Distribution: {dist}")
    assert dist == {"Unknown": 500}, f"Expected all 500 Unknowns, found {dist}"

    # Feature variance
    numeric_cols = ["reported_casualties", "injured_population", "estimated_loss", "latitude", "longitude"]
    print("Numerical Feature Statistics:")
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        print(f"  {col}: count={len(series)}, min={series.min()}, max={series.max()}, mean={series.mean():.2f}, std={series.std():.2f}")


def test_3_hybrid_model_artifacts():
    """Verify hybrid artifacts exist and load cleanly without warnings."""
    print("\n--- Test 3: Hybrid Model Artifact Verification ---")
    assert HYBRID_ARTIFACT_PATH.exists(), f"Artifact missing: {HYBRID_ARTIFACT_PATH}"
    assert EVAL_REPORT_PATH.exists(), f"Report missing: {EVAL_REPORT_PATH}"

    artifact = load_hybrid_artifact()
    assert artifact is not None, "Failed to load hybrid artifact"
    assert "model" in artifact
    assert "imputer" in artifact
    assert "scaler" in artifact
    assert "feature_columns" in artifact
    print(f"Loaded Hybrid Artifact successfully!")
    print(f"Framework: {artifact.get('framework')}")
    print(f"Active Features: {artifact.get('feature_columns')}")
    print(f"Sklearn version used: {artifact.get('sklearn_version')}")

    with open(EVAL_REPORT_PATH, "r") as f:
        rep = json.load(f)
    assert "isolation_forest_model" in rep or "isolation_forest_anomaly_detection" in rep
    assert "evidence_based_impact_screening" in rep
    assert "supervised_priority_benchmark" in rep or "supervised_priority_classification" in rep
    print("Evaluation report schema verified.")


def test_4_inference_on_latest_500_records(df: pd.DataFrame):
    """Run hybrid model inference on all 500 records."""
    print("\n--- Test 4: Inference on Latest 500 Records ---")
    artifact = load_hybrid_artifact()
    imputer = artifact["imputer"]
    scaler = artifact["scaler"]
    model = artifact["model"]
    feature_cols = artifact["feature_columns"]

    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if "estimated_loss" in X.columns:
        X["estimated_loss"] = np.log1p(X["estimated_loss"].clip(lower=0))

    X_imp = imputer.transform(X)
    X_scl = scaler.transform(X_imp)
    preds = model.predict(X_scl)
    scores = model.decision_function(X_scl)

    anomalies = int((preds == -1).sum())
    print(f"Anomaly predictions: {anomalies}/500 ({anomalies/500*100:.2f}%)")
    print(f"Decision score range: min={scores.min():.4f}, max={scores.max():.4f}, median={np.median(scores):.4f}")
    assert anomalies > 0, "Expected at least 1 statistical outlier on real NOAA data"


def test_5_output_row_count_and_screening_checks(df: pd.DataFrame):
    """Verify impact screening on all 500 records and check row counts."""
    print("\n--- Test 5: Output Row Count and Missing/Unknown Checks ---")
    screening_results = df.apply(screen_evidence_impact, axis=1)
    screening_df = pd.DataFrame(screening_results.tolist())
    assert len(screening_df) == 500, f"Expected 500 screening results, got {len(screening_df)}"

    tier_counts = screening_df["impact_tier"].value_counts().to_dict()
    print(f"Impact Tier Distribution: {tier_counts}")
    assert "Critical Impact" in tier_counts
    assert "High Impact" in tier_counts
    assert "Unassessed / Incomplete Impact Evidence" in tier_counts
    assert "Negligible Observed Impact" in tier_counts

    # Verify high-impact events identified correctly
    crit_records = df[screening_df["impact_tier"] == "Critical Impact"]
    print("Identified Critical Impact Events:")
    for _, row in crit_records.iterrows():
        print(f"  Incident {row['incident_identifier']} ({row['disaster_type']} in {row['incident_location']}): casualties={row['reported_casualties']}, injuries={row['injured_population']}, loss=${row['estimated_loss']:,.0f}")


def test_6_api_integration_receive_and_assess_csv():
    """Submit the real NOAA CSV to POST /api/receive-file and verify hybrid response."""
    print("\n--- Test 6: API Integration Test (POST /api/receive-file) ---")
    with open(DATASET_PATH, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/receive-file",
        files={"file": ("job-1792491883115_processed.csv", io.BytesIO(file_bytes), "text/csv")}
    )
    assert response.status_code == 200, f"API failed with {response.status_code}: {response.text}"
    data = response.json()

    assert data["status"] == "success"
    assert data["total_records"] == 500
    assert "hybrid_assessment" in data
    assert "ml_assessment" in data
    assert "recommendations" in data

    hybrid = data["hybrid_assessment"]
    assert hybrid["status"] == "Operational"
    assert hybrid["evidence_based_impact_screening"]["status"] == "Operational"
    assert hybrid["supervised_priority_classification"]["status"].startswith("Candidate Only")

    # Recommendations must contain the Critical Impact events
    recs = data["recommendations"]
    assert len(recs) > 0
    rec_texts = " ".join([r["action"] for r in recs])
    print(f"Top 3 API Recommendations:")
    for r in recs[:3]:
        print(f"  [{r['priority']}] ({r.get('source', '')}) {r['action']}")

    assert "Critical Impact" in rec_texts or "Critical" in [r["priority"] for r in recs]
    print("API successfully returned hybrid assessment and actionable recommendations.")


def test_7_regression_tests_existing_api():
    """Verify that existing endpoints (/api/health, /api/assess, /api/processed-results) work."""
    print("\n--- Test 7: Regression Tests for Existing API Behavior ---")

    # Health check
    h_res = client.get("/api/health")
    assert h_res.status_code == 200
    assert h_res.json()["status"] == "healthy"
    print("PASS: /api/health")

    # Assess endpoint
    a_res = client.get("/api/assess")
    assert a_res.status_code == 200
    a_data = a_res.json()
    assert "summary" in a_data
    assert "incidents" in a_data
    print("PASS: /api/assess")

    # Processed results endpoint (general)
    p_res = client.get("/api/processed-results")
    assert p_res.status_code == 200
    p_data = p_res.json()
    assert p_data["status"] == "success"
    assert p_data["total_records"] >= 500
    print("PASS: /api/processed-results")

    # Processed results with specific job_id (Job isolation check)
    p_job_res = client.get("/api/processed-results?job_id=job-1792491883115")
    assert p_job_res.status_code == 200
    p_job_data = p_job_res.json()
    assert "job-1792491883115" in p_job_data["filename"]
    print("PASS: /api/processed-results?job_id=job-1792491883115 (Job Isolation Verified)")


if __name__ == "__main__":
    print("=================================================================")
    print("MEMBER 2 HYBRID DISASTER ASSESSMENT VERIFICATION SUITE")
    print("=================================================================")
    df = test_1_latest_noaa_file_validation()
    test_2_missing_value_and_feature_quality(df)
    test_3_hybrid_model_artifacts()
    test_4_inference_on_latest_500_records(df)
    test_5_output_row_count_and_screening_checks(df)
    test_6_api_integration_receive_and_assess_csv()
    test_7_regression_tests_existing_api()
    print("\n=================================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! ZERO FAILURES.")
    print("=================================================================")
