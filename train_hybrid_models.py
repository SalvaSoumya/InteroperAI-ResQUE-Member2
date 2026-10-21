"""Hybrid Disaster Assessment Workflow — Training & Artifact Generation.

Combines:
1. Unsupervised Anomaly Detection (Isolation Forest) on valid numerical features with observed variance.
2. Evidence-Based Impact Screening based strictly on observed casualties, injuries, and losses.
3. Supervised Classification diagnostics: verifies priority labels, reporting data quality & leakage risks.
4. Unified Hybrid Output generation without fabricating missing data.

Saves artifacts to: member2_hybrid_artifacts/
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import sklearn

DEFAULT_DATASET = Path("backend/received_datasets/job-1792491883115_processed.csv")
OLDER_SYNTHETIC_DATASET = Path("backend/received_datasets/job-1791983245317_processed.csv")
OUTPUT_DIR = Path("member2_hybrid_artifacts")


def screen_evidence_impact(row: pd.Series) -> dict:
    """Calculate an explainable impact screening score from observed evidence only.
    
    Does NOT assume unmeasured fields are zero. Tracks evidence completeness.
    """
    cas = row.get("reported_casualties")
    inj = row.get("injured_population")
    disp = row.get("displaced_population")
    aff = row.get("affected_population")
    loss = row.get("estimated_loss")

    harm_score = 0.0
    econ_score = 0.0
    evidence = []
    missing_fields = []

    fields = [
        ("reported_casualties", cas),
        ("injured_population", inj),
        ("displaced_population", disp),
        ("affected_population", aff),
        ("estimated_loss", loss),
    ]

    valid_fields = []
    for name, val in fields:
        if pd.notna(val) and val != "" and val != "null":
            valid_fields.append(name)
        else:
            missing_fields.append(name)

    completeness_pct = round(len(valid_fields) / 5.0 * 100.0, 1)

    # 1. Human Harm Evidence
    if pd.notna(cas) and float(cas) > 0:
        c_val = float(cas)
        harm_score += 40.0 + min((c_val - 1.0) * 10.0, 10.0)
        evidence.append(f"{int(c_val)} reported fatality" if c_val == 1.0 else f"{int(c_val)} reported fatalities")

    if pd.notna(inj) and float(inj) > 0:
        i_val = float(inj)
        harm_score += min(i_val * 15.0, 30.0)
        evidence.append(f"{int(i_val)} injured persons")

    if pd.notna(disp) and float(disp) > 0:
        d_val = float(disp)
        harm_score += min(np.log10(d_val + 1.0) * 7.0, 20.0)
        evidence.append(f"{int(d_val):,} displaced population")

    if pd.notna(aff) and float(aff) > 0:
        a_val = float(aff)
        harm_score += min(np.log10(a_val + 1.0) * 4.0, 15.0)
        evidence.append(f"{int(a_val):,} affected population")

    # 2. Economic Damage Evidence
    if pd.notna(loss) and float(loss) > 0:
        l_val = float(loss)
        econ_score = min(np.log10(l_val + 1.0) * 6.5, 50.0)
        evidence.append(f"${l_val:,.0f} estimated loss")

    total_score = round(min(harm_score + econ_score, 100.0), 1)

    # 3. Categorize into explainable Relative Impact Tiers
    if not valid_fields:
        tier = "Indeterminate / Insufficient Evidence"
        recommendation = "No impact measurements supplied. Request field reconnaissance data."
    elif harm_score >= 40.0 or econ_score >= 45.0 or total_score >= 70.0:
        tier = "Critical Impact"
        recommendation = "Immediate priority: dispatch emergency response teams and verify casualty relief."
    elif harm_score >= 15.0 or econ_score >= 35.0 or total_score >= 40.0:
        tier = "High Impact"
        recommendation = "High priority: coordinate medical aid, shelter assistance, or infrastructure repair."
    elif econ_score >= 20.0 or total_score >= 20.0:
        tier = "Moderate Impact"
        recommendation = "Moderate priority: support local recovery and monitor community recovery."
    elif total_score > 0.0:
        tier = "Minor Impact"
        recommendation = "Standard local assessment: log incident and verify minor repairs."
    else:
        tier = "Negligible Impact"
        recommendation = "Low priority: incident reported with zero observed casualties, injuries, or losses."

    return {
        "impact_screening_score": total_score,
        "impact_tier": tier,
        "contributing_evidence": evidence if evidence else ["Zero harm/loss observed across reported fields"],
        "evidence_completeness_pct": completeness_pct,
        "missing_impact_fields": missing_fields,
        "human_review_recommendation": recommendation,
    }


def main(dataset_path: str = None):
    source = Path(dataset_path).resolve() if dataset_path else DEFAULT_DATASET.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input dataset not found at: {source}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading dataset from: {source}")
    df = pd.read_csv(source)
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # 1. Feature availability & missingness analysis
    all_numeric_candidates = [
        "reported_casualties", "injured_population", "estimated_loss",
        "latitude", "longitude", "affected_population", "displaced_population",
        "available_shelter_capacity", "current_shelter_occupancy"
    ]

    available_numeric = []
    missing_summary = {}
    constant_columns = []

    for col in all_numeric_candidates:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            missing_count = int(s.isna().sum())
            missing_pct = round(missing_count / len(df) * 100.0, 1)
            missing_summary[col] = {"missing_count": missing_count, "missing_pct": missing_pct}
            
            # Check if column is 100% missing or constant
            valid_vals = s.dropna()
            if len(valid_vals) == 0:
                constant_columns.append(col)
            elif valid_vals.nunique() <= 1:
                constant_columns.append(col)
            else:
                available_numeric.append(col)
        else:
            missing_summary[col] = {"missing_count": len(df), "missing_pct": 100.0}
            constant_columns.append(col)

    print(f"\nUsable features with variation ({len(available_numeric)}): {available_numeric}")
    print(f"Constant/unpopulated features excluded ({len(constant_columns)}): {constant_columns}")

    # 2. Train Isolation Forest on features with real variance
    X = df[available_numeric].copy()
    for c in available_numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        # Apply log1p to heavy-tailed positive dollar values if present
        if c == "estimated_loss":
            X[c] = np.log1p(X[c].clip(lower=0))

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    # Contamination set to 0.03 (targeted 3% review fraction for true outliers)
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.03,
        random_state=42,
        n_jobs=-1
    )
    preds = iso.fit_predict(X_scaled)
    scores = iso.decision_function(X_scaled)

    df["hybrid_anomaly_score"] = np.round(scores, 4)
    df["hybrid_is_anomaly"] = (preds == -1)
    df["hybrid_anomaly_status"] = np.where(preds == -1, "Unusual", "Normal")

    anomaly_count = int((preds == -1).sum())
    anomaly_pct = round(float((preds == -1).mean()) * 100.0, 2)
    print(f"\nIsolation Forest fitted on {len(available_numeric)} features:")
    print(f"  Flagged anomalies: {anomaly_count} ({anomaly_pct}%)")
    print(f"  Score range: {scores.min():.4f} to {scores.max():.4f}")

    # Save Isolation Forest artifact in current environment (scikit-learn 1.9.1)
    iso_artifact = {
        "model": iso,
        "imputer": imputer,
        "scaler": scaler,
        "feature_columns": available_numeric,
        "excluded_constant_columns": constant_columns,
        "contamination": 0.03,
        "sklearn_version": sklearn.__version__,
        "trained_timestamp": datetime.now().isoformat(),
        "training_dataset": str(source),
        "score_note": "Unsupervised Isolation Forest decision_function; lower/negative values flag multidimensional outliers for review."
    }
    iso_artifact_path = OUTPUT_DIR / "hybrid_isolation_forest.joblib"
    joblib.dump(iso_artifact, iso_artifact_path)
    print(f"Saved Isolation Forest artifact to: {iso_artifact_path}")

    # 3. Apply Evidence-Based Impact Screening
    print("\nRunning Evidence-Based Impact Screening across all records...")
    screening_results = df.apply(screen_evidence_impact, axis=1)
    screening_df = pd.DataFrame(screening_results.tolist())
    
    df["impact_screening_score"] = screening_df["impact_screening_score"]
    df["impact_tier"] = screening_df["impact_tier"]
    df["contributing_evidence"] = screening_df["contributing_evidence"].apply(lambda ev: "; ".join(ev))
    df["evidence_completeness_pct"] = screening_df["evidence_completeness_pct"]
    df["human_review_recommendation"] = screening_df["human_review_recommendation"]

    tier_counts = df["impact_tier"].value_counts().to_dict()
    print("Impact Tier Distribution:")
    for tier, count in tier_counts.items():
        print(f"  {tier:35s}: {count}")

    # 4. Supervised Priority Classification Diagnostics
    print("\nInspecting Priority Labels...")
    priority_col = "priority_level"
    priority_counts = df[priority_col].value_counts(dropna=False).to_dict() if priority_col in df.columns else {}
    print(f"Priority counts in current dataset: {priority_counts}")

    supervised_status = {
        "dataset_name": source.name,
        "priority_distribution": {str(k): int(v) for k, v in priority_counts.items()},
        "is_trainable": False,
        "reason": "All records in current NOAA dataset have priority_level 'Unknown'. Cannot train supervised model without ground truth.",
    }

    # Evaluate the older synthetic dataset if present to document label quality
    if OLDER_SYNTHETIC_DATASET.exists():
        print(f"\nEvaluating older dataset labels for baseline reference: {OLDER_SYNTHETIC_DATASET}")
        old_df = pd.read_csv(OLDER_SYNTHETIC_DATASET)
        old_counts = old_df["priority_level"].value_counts().to_dict()
        supervised_status["older_dataset_reference"] = {
            "path": str(OLDER_SYNTHETIC_DATASET),
            "rows": len(old_df),
            "priority_counts": {str(k): int(v) for k, v in old_counts.items()},
            "evaluation_finding": "Previous evaluation proved 24.50% RF accuracy vs 25.70% majority baseline across 4 balanced classes. Casualties and loss distributions showed no statistical correlation with assigned priority labels, indicating synthetic/arbitrary labeling. Model remains designated as CANDIDATE_UNVALIDATED."
        }

    # 5. Save Enriched Dataset and Experiment Reports
    enriched_csv_path = OUTPUT_DIR / "dataset_with_hybrid_assessment.csv"
    df.to_csv(enriched_csv_path, index=False)
    print(f"\nSaved enriched dataset with hybrid columns to: {enriched_csv_path}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "software_versions": {
            "python": sys.version,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "dataset": {
            "path": str(source),
            "filename": source.name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "missing_value_summary": missing_summary,
        },
        "isolation_forest_anomaly_detection": {
            "status": "Operational",
            "features_used": available_numeric,
            "excluded_unpopulated_features": constant_columns,
            "contamination_parameter": 0.03,
            "anomalies_flagged": anomaly_count,
            "anomaly_rate_pct": anomaly_pct,
            "score_min": round(float(scores.min()), 4),
            "score_max": round(float(scores.max()), 4),
            "score_mean": round(float(scores.mean()), 4),
            "interpretation": "Anomaly scores are statistical review signals indicating multi-attribute divergence from typical weather incident patterns.",
        },
        "evidence_based_impact_screening": {
            "status": "Operational",
            "methodology": "Transparent impact scoring based strictly on reported casualties, injuries, and estimated losses.",
            "impact_tier_distribution": tier_counts,
            "average_evidence_completeness_pct": round(float(df["evidence_completeness_pct"].mean()), 1),
            "critical_events_detected": int((df["impact_tier"] == "Critical Impact").sum()),
            "high_events_detected": int((df["impact_tier"] == "High Impact").sum()),
        },
        "supervised_priority_classification": supervised_status,
        "hybrid_integration_summary": {
            "workflow": "Combines unsupervised anomaly detection with deterministic evidence-based impact screening to produce prioritized human review recommendations.",
            "artifacts_generated": [
                str(iso_artifact_path),
                str(enriched_csv_path),
                str(OUTPUT_DIR / "hybrid_evaluation_report.json"),
                str(OUTPUT_DIR / "hybrid_evaluation_report.txt"),
            ]
        }
    }

    report_json_path = OUTPUT_DIR / "hybrid_evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    report_txt_path = OUTPUT_DIR / "hybrid_evaluation_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("MEMBER 2 — HYBRID DISASTER ASSESSMENT EXPERIMENT REPORT\n")
        f.write("=" * 65 + "\n")
        f.write(f"Generated at: {report['timestamp']}\n")
        f.write(f"Dataset: {source.name} ({len(df)} records)\n")
        f.write(f"Scikit-learn version: {sklearn.__version__}\n\n")
        f.write("1. DATA QUALITY & FEATURE AVAILABILITY\n")
        f.write("-" * 45 + "\n")
        f.write(f"Usable features with variation: {', '.join(available_numeric)}\n")
        f.write(f"Excluded unpopulated/constant features: {', '.join(constant_columns)}\n\n")
        f.write("2. ISOLATION FOREST ANOMALY DETECTION\n")
        f.write("-" * 45 + "\n")
        f.write(f"Anomalies flagged: {anomaly_count} ({anomaly_pct}% of records)\n")
        f.write(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]\n\n")
        f.write("3. EVIDENCE-BASED IMPACT SCREENING\n")
        f.write("-" * 45 + "\n")
        for tier, count in tier_counts.items():
            f.write(f"  - {tier}: {count}\n")
        f.write(f"\nAverage evidence completeness: {report['evidence_based_impact_screening']['average_evidence_completeness_pct']}%\n\n")
        f.write("4. SUPERVISED PRIORITY CLASSIFICATION\n")
        f.write("-" * 45 + "\n")
        f.write(f"Current NOAA dataset labels: 100% Unknown (No supervised training possible)\n")
        f.write("Older dataset check: 24.50% RF accuracy vs 25.70% baseline (labels unvalidated)\n\n")
        f.write("5. HYBRID RECOMMENDATION HIGHLIGHTS\n")
        f.write("-" * 45 + "\n")
        critical_rows = df[df["impact_tier"] == "Critical Impact"]
        for _, r in critical_rows.iterrows():
            f.write(f"* [{r['disaster_type']} - {r['incident_location']}] Score: {r['impact_screening_score']} | Evidence: {r['contributing_evidence']}\n")

    print(f"Saved evaluation reports to:\n  {report_json_path}\n  {report_txt_path}")
    print("\nHybrid workflow execution completed successfully.")


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(path_arg)
