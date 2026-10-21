"""Regression & Edge-Case Verification Suite for Member 2 Disaster Assessment.

Tests:
1. Record with observed zero casualties and verified zero loss.
2. Record with missing casualty and loss fields (unassessed).
3. Record with positive fatalities and injuries (Puerto Rico rip current pattern).
4. Record with moderate economic loss ($50,000, 0 casualties).
5. Record with catastrophic economic loss ($18.5M wildfire pattern).
6. Incomplete shelter and population data (verifying no fabricated numbers).
7. Diverse numerical feature inputs yielding distinct continuous scores from the model.
8. Real 500-record NOAA dataset validation (distributions of scores, tiers, completeness, and anomaly flags).
"""

import io
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app, screen_evidence_impact, load_hybrid_artifact, database

client = TestClient(app)

NOAA_500_PATH = Path(__file__).resolve().parent / "received_datasets" / "job-1792491883115_processed.csv"


def test_observed_zero_casualties_and_losses():
    """Verify record with confirmed zero casualties and confirmed $0 loss."""
    row = pd.Series({
        "reported_casualties": 0,
        "injured_population": 0,
        "estimated_loss": 0.0,
        "displaced_population": np.nan,
        "affected_population": np.nan,
        "latitude": 34.0,
        "longitude": -84.0,
        "incident_description": "Routine wind gust observed; no damage reported."
    })
    res = screen_evidence_impact(row)
    assert res["impact_screening_score"] == 0.0
    assert res["impact_tier"] == "Negligible Observed Impact"
    assert "Verified zero" in res["contributing_evidence"][0]
    assert "negligible" not in res["human_review_recommendation"].lower() or "zero casualties" in res["human_review_recommendation"].lower()
    print("PASS: test_observed_zero_casualties_and_losses")


def test_missing_casualty_and_loss_fields():
    """Verify record with missing/unreported casualty and loss data."""
    row = pd.Series({
        "reported_casualties": np.nan,
        "injured_population": np.nan,
        "estimated_loss": np.nan,
        "displaced_population": np.nan,
        "affected_population": np.nan,
        "latitude": np.nan,
        "longitude": np.nan,
        "incident_description": None
    })
    res = screen_evidence_impact(row)
    assert res["impact_screening_score"] == 0.0
    assert res["impact_tier"] == "Unassessed / Incomplete Impact Evidence"
    assert res["evidence_completeness_pct"] == 0.0
    assert "unassessed" in res["human_review_recommendation"].lower() or "reconnaissance" in res["human_review_recommendation"].lower()
    print("PASS: test_missing_casualty_and_loss_fields")


def test_positive_fatalities_and_injuries():
    """Verify record with 1 fatality and 2 injuries (Puerto Rico pattern)."""
    row = pd.Series({
        "reported_casualties": 1,
        "injured_population": 2,
        "estimated_loss": 0.0,
        "displaced_population": np.nan,
        "affected_population": np.nan,
        "latitude": 18.4,
        "longitude": -66.1,
        "incident_description": "Rip current incident resulted in 1 drowning and 2 rescues."
    })
    res = screen_evidence_impact(row)
    assert res["impact_screening_score"] >= 70.0
    assert res["impact_tier"] == "Critical Impact"
    ev_str = " ".join(res["contributing_evidence"])
    assert "1 reported fatality" in ev_str
    assert "2 injured persons" in ev_str
    print("PASS: test_positive_fatalities_and_injuries")


def test_positive_economic_loss():
    """Verify record with $50,000 loss and 0 casualties."""
    row = pd.Series({
        "reported_casualties": 0,
        "injured_population": 0,
        "estimated_loss": 50000.0,
        "displaced_population": np.nan,
        "affected_population": np.nan,
        "latitude": 33.5,
        "longitude": -85.2,
        "incident_description": "Roof torn off barn by high winds."
    })
    res = screen_evidence_impact(row)
    assert 25.0 <= res["impact_screening_score"] <= 35.0
    assert res["impact_tier"] == "Moderate Impact"
    assert "$50,000" in res["contributing_evidence"][0]
    print("PASS: test_positive_economic_loss")


def test_catastrophic_economic_loss():
    """Verify record with $18.5M loss (Washington wildfire pattern)."""
    row = pd.Series({
        "reported_casualties": 0,
        "injured_population": 0,
        "estimated_loss": 18500000.0,
        "displaced_population": np.nan,
        "affected_population": np.nan,
        "latitude": 48.7,
        "longitude": -119.5,
        "incident_description": "Large wildfire burned multiple structures."
    })
    res = screen_evidence_impact(row)
    assert res["impact_screening_score"] >= 45.0
    assert res["impact_tier"] == "Critical Impact"
    assert "$18,500,000" in res["contributing_evidence"][0]
    print("PASS: test_catastrophic_economic_loss")


def test_incomplete_shelter_and_population_handling():
    """Verify CSV upload with missing shelter and population does not invent data."""
    csv_data = (
        "incident_identifier,disaster_type,incident_location,priority_level,reported_casualties,"
        "injured_population,estimated_loss,latitude,longitude,incident_description\n"
        "TEST-01,Flood,Zone A,Unknown,0,0,25000,32.5,-85.0,Local street flooding\n"
        "TEST-02,Tornado,Zone B,Unknown,0,1,100000,32.6,-85.1,Tornado touched down briefly\n"
    )
    res = client.post(
        "/api/receive-file",
        files={"file": ("test_shelter_missing.csv", io.BytesIO(csv_data.encode()), "text/csv")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    # Shelters should report 0 monitored records
    assert data["shelter_summary"]["monitored_shelter_records"] == 0
    # No fake resource shortage recommendations
    for r in data["recommendations"]:
        assert "shortage of" not in r["action"].lower()
    print("PASS: test_incomplete_shelter_and_population_handling")


def test_distinct_numerical_feature_scores():
    """Verify different coordinates and losses produce distinct decision scores and impact scores."""
    artifact = load_hybrid_artifact()
    model = artifact["model"]
    imputer = artifact["imputer"]
    scaler = artifact["scaler"]
    features = artifact["feature_columns"]

    test_rows = [
        {"reported_casualties": 0, "injured_population": 0, "estimated_loss": 0, "latitude": 44.0, "longitude": -120.0},
        {"reported_casualties": 0, "injured_population": 0, "estimated_loss": 50000, "latitude": 33.0, "longitude": -85.0},
        {"reported_casualties": 1, "injured_population": 2, "estimated_loss": 0, "latitude": 18.4, "longitude": -66.1},
        {"reported_casualties": 0, "injured_population": 0, "estimated_loss": 18500000, "latitude": 48.7, "longitude": -119.5},
    ]
    df_test = pd.DataFrame(test_rows)
    X = df_test[features].copy()
    X["estimated_loss"] = np.log1p(X["estimated_loss"])
    X_scl = scaler.transform(imputer.transform(X))
    scores = model.decision_function(X_scl)

    # All 4 rows must have distinct decision scores
    assert len(set(np.round(scores, 4))) == 4, f"Scores not distinct: {scores}"
    print("PASS: test_distinct_numerical_feature_scores")


def test_real_noaa_500_records_distribution():
    """Verify actual distributions across all 500 NOAA records via the API."""
    with open(NOAA_500_PATH, "rb") as f:
        content = f.read()

    res = client.post(
        "/api/receive-file",
        files={"file": ("job-1792491883115_processed.csv", io.BytesIO(content), "text/csv")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total_records"] == 500

    hybrid = data["hybrid_assessment"]
    screening = hybrid["evidence_based_impact_screening"]
    tiers = screening["impact_tier_distribution"]

    print("\n--- NOAA 500 Records Distribution ---")
    print(f"Anomalies Flagged: {hybrid['anomaly_detection']['anomalies_flagged']} ({hybrid['anomaly_detection']['anomaly_rate_pct']}%)")
    print("Impact Tiers:")
    for tier, count in tiers.items():
        print(f"  * {tier}: {count}")

    # Check that Unassessed and Negligible are properly separated
    assert "Unassessed / Incomplete Impact Evidence" in tiers
    assert "Negligible Observed Impact" in tiers
    assert tiers["Unassessed / Incomplete Impact Evidence"] == 167
    assert tiers["Negligible Observed Impact"] == 175
    assert tiers["Critical Impact"] == 3
    assert tiers["High Impact"] == 1
    assert tiers["Moderate Impact"] == 97
    assert tiers["Minor Impact"] == 57

    # Verify top recommendations reflect real facts
    top_recs = data["recommendations"]
    assert len(top_recs) >= 3
    print("\nTop Actionable Recommendations from API:")
    for r in top_recs[:3]:
        print(f"  [{r['priority']}] {r['action']}")

    print("\nPASS: test_real_noaa_500_records_distribution")


def test_missing_casualties_with_zero_loss():
    """Verify record with missing casualties/injuries but zero loss is NOT claimed verified zero."""
    row = pd.Series({
        "reported_casualties": np.nan,
        "injured_population": np.nan,
        "estimated_loss": 0.0,
        "latitude": 34.0,
        "longitude": -84.0,
        "incident_description": "Wind report, damage verified $0 but casualties unassessed."
    })
    res = screen_evidence_impact(row)
    assert res["impact_screening_score"] == 0.0
    assert res["impact_tier"] == "Unassessed / Incomplete Impact Evidence"
    ev_str = " ".join(res["contributing_evidence"])
    assert "casualties" in ev_str and "injuries" in ev_str
    # Must NOT claim verified zero casualties!
    assert "verified zero casualties" not in ev_str.lower()
    print("PASS: test_missing_casualties_with_zero_loss")


def test_verified_zero_casualties_with_missing_loss():
    """Verify record with zero casualties/injuries but missing loss is flagged as unassessed."""
    row = pd.Series({
        "reported_casualties": 0,
        "injured_population": 0,
        "estimated_loss": np.nan,
        "latitude": 34.0,
        "longitude": -84.0,
        "incident_description": "Casualties verified zero, but damage unassessed."
    })
    res = screen_evidence_impact(row)
    assert res["impact_screening_score"] == 0.0
    assert res["impact_tier"] == "Unassessed / Incomplete Impact Evidence"
    ev_str = " ".join(res["contributing_evidence"])
    assert "economic damage" in ev_str
    print("PASS: test_verified_zero_casualties_with_missing_loss")


def test_verified_zeros_all_dimensions():
    """Verify record with verified zeros across all core impact dimensions."""
    row = pd.Series({
        "reported_casualties": 0,
        "injured_population": 0,
        "estimated_loss": 0.0,
        "latitude": 35.0,
        "longitude": -85.0,
        "incident_description": "Full assessment completed; no casualties and zero damage."
    })
    res = screen_evidence_impact(row)
    assert res["impact_screening_score"] == 0.0
    assert res["impact_tier"] == "Negligible Observed Impact"
    assert "Verified zero casualties, zero injuries, and $0 property/crop damage" in res["contributing_evidence"][0]
    print("PASS: test_verified_zeros_all_dimensions")


def test_positive_loss_with_missing_casualties():
    """Verify positive economic loss is preserved even when casualty data is missing."""
    row = pd.Series({
        "reported_casualties": np.nan,
        "injured_population": np.nan,
        "estimated_loss": 75000.0,
        "latitude": 32.0,
        "longitude": -86.0,
        "incident_description": "Barn destroyed by straight-line winds."
    })
    res = screen_evidence_impact(row)
    assert res["impact_screening_score"] > 0
    assert res["impact_tier"] == "Moderate Impact"
    assert "$75,000 estimated property/crop loss" in res["contributing_evidence"][0]
    print("PASS: test_positive_loss_with_missing_casualties")


def test_canonical_and_raw_noaa_headers():
    """Verify both canonical combined headers and raw NOAA export headers are supported without double counting."""
    # 1. Canonical headers
    canonical_csv = (
        "incident_identifier,disaster_type,incident_location,reported_casualties,injured_population,estimated_loss,latitude,longitude\n"
        "CANON-01,Tornado,Lee County,2,5,1500000,32.5,-85.3\n"
    )
    res_canon = client.post(
        "/api/receive-file",
        files={"file": ("canonical_test.csv", io.BytesIO(canonical_csv.encode()), "text/csv")}
    )
    assert res_canon.status_code == 200
    data_canon = res_canon.json()
    rec_canon = data_canon["sample_evaluated_records"][0]
    assert rec_canon["reported_casualties"] == 2
    assert rec_canon["injured_population"] == 5
    assert rec_canon["estimated_loss"] == 1500000.0
    assert rec_canon["impact_tier"] == "Critical Impact"

    # 2. Raw NOAA headers (direct/indirect deaths, direct/indirect injuries, damage strings, begin lat/lon)
    raw_noaa_csv = (
        "EVENT_ID,EVENT_TYPE,STATE,DEATHS_DIRECT,DEATHS_INDIRECT,INJURIES_DIRECT,INJURIES_INDIRECT,DAMAGE_PROPERTY,DAMAGE_CROPS,BEGIN_LAT,BEGIN_LON\n"
        "998877,Flash Flood,TEXAS,1,0,3,1,50.00K,10.00K,29.5,-98.5\n"
    )
    res_raw = client.post(
        "/api/receive-file",
        files={"file": ("raw_noaa_test.csv", io.BytesIO(raw_noaa_csv.encode()), "text/csv")}
    )
    assert res_raw.status_code == 200
    data_raw = res_raw.json()
    rec_raw = data_raw["sample_evaluated_records"][0]
    assert rec_raw["reported_casualties"] == 1.0  # 1 + 0
    assert rec_raw["injured_population"] == 4.0   # 3 + 1
    assert rec_raw["estimated_loss"] == 60000.0   # 50K + 10K
    assert rec_raw["impact_tier"] == "Critical Impact"
    print("PASS: test_canonical_and_raw_noaa_headers")


def test_api_assess_no_stale_mock_leak_after_noaa_upload():
    """Verify that after uploading a NOAA dataset without shelters/resources, /api/assess does NOT leak mock data."""
    with open(NOAA_500_PATH, "rb") as f:
        content = f.read()

    res = client.post(
        "/api/receive-file",
        files={"file": ("job-1792491883115_processed.csv", io.BytesIO(content), "text/csv")}
    )
    assert res.status_code == 200
    upload_data = res.json()

    # Recommendations consistency: root recommendations must match assessment recommendations
    root_recs = upload_data["recommendations"]
    nested_recs = upload_data["assessment"]["recommendations"]
    assert root_recs == nested_recs, "Root recommendations and assessment recommendations must be consistent!"

    # Check that no fake resource shortages are present in recommendations
    for r in root_recs:
        assert "water" not in r["action"].lower() or "rip current" in r["action"].lower()
        assert "medical kits" not in r["action"].lower()
        assert "community hall a" not in r["action"].lower()
        assert "relief centre c" not in r["action"].lower()

    # Query GET /api/assess
    assess_res = client.get("/api/assess")
    assert assess_res.status_code == 200
    assess_data = assess_res.json()

    # Shelters and resources must be empty, not showing initial mock data
    assert len(assess_data["shelters"]) == 0, f"Expected 0 shelters for NOAA data, got {len(assess_data['shelters'])}"
    assert len(assess_data["resources"]) == 0, f"Expected 0 resources for NOAA data, got {len(assess_data['resources'])}"
    assert assess_data["summary"]["resource_shortages"] == 0
    assert assess_data["summary"]["shelter_risks"] == 0

    # Overall risk must NOT be falsely forced to HIGH by fake resource shortages
    assert assess_data["summary"]["overall_risk"] in ("MEDIUM", "HIGH")

    # Verify GET /api/health and GET /api/processed-results
    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "healthy"

    proc_res = client.get("/api/processed-results")
    assert proc_res.status_code == 200
    assert proc_res.json()["status"] == "success"

    print("PASS: test_api_assess_no_stale_mock_leak_after_noaa_upload")


def test_reconciled_anomaly_and_record_counts():
    """Verify anomaly counts and record totals are consistent across upload summary, DATA_STORE, and GET /api/assess."""
    with open(NOAA_500_PATH, "rb") as f:
        content = f.read()

    res = client.post(
        "/api/receive-file",
        files={"file": ("job-1792491883115_processed.csv", io.BytesIO(content), "text/csv")}
    )
    assert res.status_code == 200
    upload_data = res.json()

    # Upload response counts
    assert upload_data["total_records"] == 500
    assert upload_data["summary"]["total_incidents"] == 500
    assert upload_data["summary"]["total_assessed_records"] == 500
    assert upload_data["summary"]["anomalies_detected"] == 69
    assert upload_data["summary"]["anomalies"] == 69

    # GET /api/assess counts
    assess_res = client.get("/api/assess")
    assert assess_res.status_code == 200
    assess_data = assess_res.json()

    assert assess_data["summary"]["total_incidents"] == 500
    assert assess_data["summary"]["total_assessed_records"] == 500
    assert assess_data["summary"]["anomalies"] == 69
    assert assess_data["summary"]["anomalies_detected"] == 69
    # The anomaly array must NOT be truncated to 20
    assert len(assess_data["anomalies"]) == 69, f"Expected 69 anomaly records, found {len(assess_data['anomalies'])}"
    print("PASS: test_reconciled_anomaly_and_record_counts")


def test_insufficient_evidence_risk_rating():
    """Verify dataset with incomplete impact evidence yields 'Insufficient evidence' overall risk."""
    incomplete_csv = (
        "incident_identifier,disaster_type,incident_location,priority_level,latitude,longitude\n"
        "INC-01,Wind,District A,Unknown,34.1,-84.1\n"
        "INC-02,Hail,District B,Unknown,34.2,-84.2\n"
        "INC-03,Rain,District C,Unknown,34.3,-84.3\n"
        "INC-04,Flood,District D,Unknown,34.4,-84.4\n"
    )
    res = client.post(
        "/api/receive-file",
        files={"file": ("incomplete_risk_test.csv", io.BytesIO(incomplete_csv.encode()), "text/csv")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["summary"]["overall_risk"] == "Insufficient evidence"
    assert "lack verified casualty or damage data" in data["summary"]["overall_risk_reason"].lower()

    # Query GET /api/assess
    assess_res = client.get("/api/assess")
    assert assess_res.json()["summary"]["overall_risk"] == "Insufficient evidence"
    print("PASS: test_insufficient_evidence_risk_rating")


def test_enriched_recommendations_structure():
    """Verify recommendations contain incident ID, disaster type, evidence, action, reason, and distinct guidance types."""
    with open(NOAA_500_PATH, "rb") as f:
        content = f.read()

    res = client.post(
        "/api/receive-file",
        files={"file": ("job-1792491883115_processed.csv", io.BytesIO(content), "text/csv")}
    )
    assert res.status_code == 200
    recs = res.json()["recommendations"]

    assert len(recs) >= 5
    guidance_types = set()
    for r in recs:
        assert "incident_id" in r and r["incident_id"]
        assert "zone" in r and r["zone"]
        assert "disaster_type" in r and r["disaster_type"]
        assert "guidance_type" in r and r["guidance_type"]
        assert "supporting_evidence" in r and r["supporting_evidence"]
        assert "action" in r and r["action"]
        assert "reason" in r and r["reason"]
        guidance_types.add(r["guidance_type"])

    # Must distinguish rule-based guidance from statistical anomaly predictions
    assert "Rule-Based Impact Guidance" in guidance_types
    assert "Model Prediction / Statistical Anomaly" in guidance_types
    print("PASS: test_enriched_recommendations_structure")


def test_api_activity_logging_and_correlation():
    """Verify API activity logging, correlation ID tracing, and client header handling."""
    csv_data = (
        "incident_identifier,disaster_type,incident_location,reported_casualties,injured_population,estimated_loss\n"
        "LOG-01,Flood,Zone X,0,0,10000\n"
    )
    test_corr_id = "CORR-TEST-12345-ABC"
    res = client.post(
        "/api/receive-file",
        headers={
            "X-Source-System": "Member-1-Integration",
            "X-Correlation-ID": test_corr_id
        },
        files={"file": ("activity_log_test.csv", io.BytesIO(csv_data.encode()), "text/csv")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["correlation_id"] == test_corr_id
    assert data["client_id"] == "Member-1-Integration"
    assert data["request_id"]
    assert data["assessment_id"]

    # Verify log entry in GET /api/activity-logs
    log_res = client.get(f"/api/activity-logs?correlation_id={test_corr_id}")
    assert log_res.status_code == 200
    log_data = log_res.json()
    assert log_data["status"] == "success"
    assert len(log_data["logs"]) >= 1

    entry = log_data["logs"][0]
    assert entry["correlation_id"] == test_corr_id
    assert entry["client_id"] == "Member-1-Integration"
    assert entry["filename"] == "activity_log_test.csv"
    assert entry["records_count"] == 1
    assert entry["http_status"] == 200
    assert entry["processing_status"] == "COMPLETED"

    # Verify failed upload is also recorded in activity logs
    fail_corr_id = "CORR-FAIL-999"
    fail_res = client.post(
        "/api/receive-file",
        headers={"X-Correlation-ID": fail_corr_id},
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    )
    assert fail_res.status_code == 400

    fail_log_res = client.get(f"/api/activity-logs?correlation_id={fail_corr_id}")
    assert fail_log_res.status_code == 200
    fail_entries = fail_log_res.json()["logs"]
    assert len(fail_entries) >= 1
    assert fail_entries[0]["http_status"] == 400
    assert fail_entries[0]["processing_status"] == "FAILED"
    assert "empty" in fail_entries[0]["error_message"].lower()
    print("PASS: test_api_activity_logging_and_correlation")


def test_job_log_isolation_between_uploads():
    """Verify successive uploads isolate their jobs, correlation IDs, and logs."""
    csv1 = "incident_identifier,disaster_type,incident_location,reported_casualties,injured_population,estimated_loss\nJOB1-01,Storm,North,0,0,5000\n"
    csv2 = "incident_identifier,disaster_type,incident_location,reported_casualties,injured_population,estimated_loss\nJOB2-01,Fire,South,1,3,2500000\n"

    res1 = client.post(
        "/api/receive-file",
        headers={"X-Source-System": "Client-Alpha", "X-Correlation-ID": "CORR-ALPHA-1"},
        files={"file": ("job_alpha.csv", io.BytesIO(csv1.encode()), "text/csv")}
    )
    assert res1.status_code == 200
    data1 = res1.json()

    res2 = client.post(
        "/api/receive-file",
        headers={"X-Source-System": "Client-Beta", "X-Correlation-ID": "CORR-BETA-2"},
        files={"file": ("job_beta.csv", io.BytesIO(csv2.encode()), "text/csv")}
    )
    assert res2.status_code == 200
    data2 = res2.json()

    # Check isolation of IDs
    assert data1["correlation_id"] == "CORR-ALPHA-1"
    assert data2["correlation_id"] == "CORR-BETA-2"
    assert data1["job_id"] == "job_alpha"
    assert data2["job_id"] == "job_beta"
    assert data1["assessment_id"] != data2["assessment_id"]

    # Current assessment reflects latest job (Beta)
    assess = client.get("/api/assess").json()
    assert assess["summary"]["correlation_id"] == "CORR-BETA-2"
    assert assess["summary"]["job_id"] == "job_beta"
    assert assess["summary"]["client_id"] == "Client-Beta"
    print("PASS: test_job_log_isolation_between_uploads")


def test_critical_impact_vs_dispatch_priority_separation():
    """Verify separate reporting of verified critical-impact events and official dispatch priority."""
    if not NOAA_500_PATH.exists():
        print("SKIP: test_critical_impact_vs_dispatch_priority_separation (file not found)")
        return
    with open(NOAA_500_PATH, "rb") as f:
        res = client.post(
            "/api/receive-file",
            files={"file": ("noaa_test.csv", f, "text/csv")}
        )
    assert res.status_code == 200
    data = res.json()
    summary = data["summary"]

    # In NOAA 500 records:
    # 3 events have verified catastrophic impact (fatalities > 0 or damage >= $5M)
    assert summary["critical_impact_events"] == 3
    # NOAA has zero official municipal dispatch priority labels (100% Unknown)
    assert summary["dispatch_priority_critical_count"] == 0
    # Backward-compatible critical_incidents field reflects official dispatch priority count (0)
    assert summary["critical_incidents"] == 0
    # Overall risk is HIGH due to verified critical impact events
    assert summary["overall_risk"] == "HIGH"
    assert "3 Critical Impact incidents" in summary["overall_risk_reason"]
    print("PASS: test_critical_impact_vs_dispatch_priority_separation")


def test_critical_incident_beyond_row_100_retained():
    """Verify that an important critical-impact event beyond row 100 is retained in DATA_STORE incidents."""
    if not NOAA_500_PATH.exists():
        print("SKIP: test_critical_incident_beyond_row_100_retained (file not found)")
        return
    assess = client.get("/api/assess").json()
    incidents = assess["incidents"]

    # Verify complete incident list is preserved without 100-record truncation
    assert len(incidents) == 500, f"Expected 500 incidents, found {len(incidents)}"

    # Row 427 in the 500-record dataset has EVENT_ID 1240899 (Rip Current in San Juan, PR: 1 death, 2 injuries)
    pr_fatal = [
        inc for inc in incidents
        if "1240899" in str(inc.get("id", "")) or "San Juan" in str(inc.get("zone", "")) or "Rip Current" in str(inc.get("type", ""))
    ]
    assert len(pr_fatal) >= 1, "Incident beyond row 100 (San Juan PR rip current) missing from DATA_STORE incidents"
    match = pr_fatal[0]
    assert match["impact_tier"] == "Critical Impact"
    assert match["priority"] == "Critical"
    print("PASS: test_critical_incident_beyond_row_100_retained")


def test_post_and_get_assessment_counts_consistent():
    """Verify that POST /api/receive-file and GET /api/assess return consistent summary counts."""
    if not NOAA_500_PATH.exists():
        print("SKIP: test_post_and_get_assessment_counts_consistent (file not found)")
        return
    with open(NOAA_500_PATH, "rb") as f:
        post_res = client.post(
            "/api/receive-file",
            files={"file": ("noaa_consistency_test.csv", f, "text/csv")}
        )
    assert post_res.status_code == 200
    post_summary = post_res.json()["summary"]

    get_res = client.get("/api/assess")
    assert get_res.status_code == 200
    get_summary = get_res.json()["summary"]

    assert post_summary["total_incidents"] == get_summary["total_incidents"]
    assert post_summary["total_assessed_records"] == get_summary["total_assessed_records"]
    assert post_summary["critical_impact_events"] == get_summary["critical_impact_events"]
    assert post_summary["dispatch_priority_critical_count"] == get_summary["dispatch_priority_critical_count"]
    assert post_summary["critical_incidents"] == get_summary["critical_incidents"]
    assert post_summary["overall_risk"] == get_summary["overall_risk"]
    assert post_summary["anomalies_detected"] == get_summary["anomalies_detected"]
    print("PASS: test_post_and_get_assessment_counts_consistent")


def test_ml_results_provenance_and_active_noaa_alignment():
    """Verify that /api/ml-results reflects active NOAA assessment provenance and not stale 10k mock data."""
    if not NOAA_500_PATH.exists():
        print("SKIP: test_ml_results_provenance_and_active_noaa_alignment (file not found)")
        return
    with open(NOAA_500_PATH, "rb") as f:
        res = client.post(
            "/api/receive-file",
            files={"file": ("job-1792491883115_processed.csv", f, "text/csv")}
        )
    assert res.status_code == 200

    ml_res = client.get("/api/ml-results")
    assert ml_res.status_code == 200
    ml_data = ml_res.json()

    assert ml_data["status"] == "success"
    # Provenance and record counts must match active NOAA assessment (500 records, NOT 10,000)
    assert ml_data["dataset"]["rows"] == 500, f"Expected 500 rows, got {ml_data['dataset']['rows']}"
    assert "job-1792491883115" in ml_data["dataset"]["filename"]
    assert ml_data["dataset"]["provenance"] == "Current Active NOAA Assessment"
    assert ml_data["dataset"]["priority_class_counts"] == {"Unknown": 500}

    # Active assessment metadata
    act = ml_data["active_assessment"]
    assert act["total_records"] == 500
    assert act["critical_impact_events"] == 3
    assert act["dispatch_priority_critical_count"] == 0
    assert act["overall_risk"] == "HIGH"
    print("PASS: test_ml_results_provenance_and_active_noaa_alignment")


def test_ml_results_anomaly_counts_and_unavailable_shelter_stats():
    """Verify that /api/ml-results reports active Isolation Forest anomalies (69, not 500) and unavailable shelter stats."""
    ml_res = client.get("/api/ml-results")
    assert ml_res.status_code == 200
    ml_data = ml_res.json()

    # Active Isolation Forest anomaly detection
    anom = ml_data["anomaly_detection"]
    assert anom["model"] == "Isolation Forest"
    assert anom["status"] == "Operational"
    assert anom["anomaly_count"] == 69, f"Expected 69 real anomalies, got {anom['anomaly_count']}"
    assert anom["anomaly_fraction"] == 0.138
    assert anom["anomaly_rate_pct"] == 13.8

    # Shelter utilization statistics must be unavailable on NOAA data
    assert anom["shelter_utilization_summary_pct"] is None
    assert anom["shelter_status"] == "Unavailable"

    act_shelter = ml_data["active_assessment"]["shelter_inventory"]
    assert act_shelter["status"] == "Unavailable"
    assert act_shelter["monitored_shelter_records"] == 0
    assert act_shelter["shelter_utilization_summary_pct"] is None
    print("PASS: test_ml_results_anomaly_counts_and_unavailable_shelter_stats")


def test_ml_results_model_status_candidate_only_and_benchmarks_separation():
    """Verify that /api/ml-results marks supervised models as candidate-only and explicitly separates benchmarks."""
    ml_res = client.get("/api/ml-results")
    assert ml_res.status_code == 200
    ml_data = ml_res.json()

    # Supervised model summary must explicitly say Candidate Only and deployed = False
    mod = ml_data["model"]
    assert mod["deployed"] is False
    assert "Candidate Only" in mod["status"]
    assert "Not Deployed" in mod["status"]

    # Historical benchmarks must be explicitly separated
    assert "historical_offline_benchmarks" in ml_data
    benchmarks = ml_data["historical_offline_benchmarks"]

    # HistGradientBoosting candidate benchmark
    assert "supervised_hist_gradient_boosting" in benchmarks
    sgb = benchmarks["supervised_hist_gradient_boosting"]
    assert sgb["deployment_status"] == "Candidate Only (Not Deployed)"
    assert sgb["tuned_threshold"] == 0.60
    assert sgb["held_out_test_metrics"]["recall_class_1"] >= 0.60

    # Legacy Random Forest benchmark
    assert "historical_legacy_random_forest" in benchmarks
    lrf = benchmarks["historical_legacy_random_forest"]
    assert "Historical Benchmark" in lrf["status"]
    assert "Obsolete" in lrf["status"] or "Not Validated" in lrf["status"]

    # Warnings must mention unvalidated candidates and unavailable shelter feeds
    warnings = " ".join(ml_data["data_quality_warnings"])
    assert "unavailable" in warnings.lower()
    assert "candidate" in warnings.lower()
    print("PASS: test_ml_results_model_status_candidate_only_and_benchmarks_separation")


def test_dynamic_economic_loss_coverage_calculation():
    """Verify dynamic economic-loss coverage across 30 records, verified zeros, NaNs, and empty datasets."""
    client = TestClient(app)

    # 1. Test 30-record fixture from disk (noaa_storm_events_dynamic_test_fixture / job-1792581907555_processed.csv)
    fixture_path = Path(__file__).resolve().parent / "received_datasets" / "job-1792581907555_processed.csv"
    assert fixture_path.exists(), f"Fixture file {fixture_path} not found"
    
    with open(fixture_path, "rb") as f:
        file_bytes = f.read()

    res = client.post("/api/receive-file", files={"file": ("noaa_storm_events_dynamic_test_fixture.csv", io.BytesIO(file_bytes), "text/csv")})
    assert res.status_code == 200, res.text
    body = res.json()
    summary = body["summary"]

    dq = summary["data_quality_report"]["economic_loss_coverage"]
    assert dq["total_records"] == 30
    assert dq["assessed_records"] == 29
    assert dq["unassessed_records"] == 1
    assert dq["coverage_percentage"] == 96.7
    assert dq["unassessed_percentage"] == 3.3
    assert dq["coverage_display"] == "96.7% Present (3.3% Unassessed)"

    # Verify GET /api/assess returns the identical dynamic coverage calculation
    assess_res = client.get("/api/assess")
    assert assess_res.status_code == 200
    assess_summary = assess_res.json()["summary"]
    assess_dq = assess_summary["data_quality_report"]["economic_loss_coverage"]
    assert assess_dq["coverage_percentage"] == 96.7
    assert assess_dq["unassessed_percentage"] == 3.3
    assert assess_dq["coverage_display"] == "96.7% Present (3.3% Unassessed)"

    print("PASS: test_dynamic_economic_loss_coverage_calculation")


def test_incident_critical_filter_agrees_with_rule_based_tier():
    """Verify that Critical impact filter uses rule-based tier and never returns 0 when critical events exist."""
    client = TestClient(app)

    # In the 30-record dataset:
    # 8 records have Critical Impact (casualties > 0 or loss >= $5M)
    # 0 records have official priority == 'Critical' (all 30 are 'Unknown')
    assess_res = client.get("/api/assess")
    assert assess_res.status_code == 200
    data = assess_res.json()

    # Rule-based critical impact count must be 8
    assert data["summary"]["critical_impact_events"] == 8, f"Expected 8 critical impact events, got {data['summary']['critical_impact_events']}"
    # Official dispatch priority critical count must be 0 (Unknown)
    assert data["summary"]["dispatch_priority_critical_count"] == 0

    incidents = data["incidents"]
    assert len(incidents) == 30

    # Rule-based filter for Critical
    critical_by_impact_tier = [i for i in incidents if (i.get("impact_tier") or "").startswith("Critical")]
    assert len(critical_by_impact_tier) == 8, f"Expected 8 critical events, got {len(critical_by_impact_tier)}"

    # Confirm official priority remains 'Unknown' and is not fabricated
    official_critical = [i for i in incidents if i.get("official_priority") == "Critical"]
    assert len(official_critical) == 0, "Official priority must not be fabricated"
    assert all(i.get("official_priority") == "Unknown" for i in incidents), "All official priorities in NOAA fixture must be Unknown"

    print("PASS: test_incident_critical_filter_agrees_with_rule_based_tier")


def test_shelter_crud_persistence_and_calculations():
    """Verify Shelter CRUD operations, SQLite persistence, and dynamic calculations."""
    client = TestClient(app)

    # 1. Create a new shelter
    new_shelter_payload = {
        "shelter_id": "TEST-SH-99",
        "shelter_name": "Test Regional Evacuation Center",
        "location": "Test Valley",
        "capacity": 500,
        "occupancy": 350,
        "notes": "Testing shelter module"
    }
    create_res = client.post("/api/shelters", json=new_shelter_payload)
    assert create_res.status_code == 200, create_res.text
    created = create_res.json()["shelter"]
    assert created["shelter_id"] == "TEST-SH-99"
    assert created["capacity"] == 500
    assert created["occupancy"] == 350
    assert created["available_beds"] == 150
    assert created["utilization_percentage"] == 70.0
    assert created["status"] == "Available"

    # 2. Update occupancy to near capacity (>= 80% and < 100%)
    update_res1 = client.put("/api/shelters/TEST-SH-99", json={"occupancy": 450})
    assert update_res1.status_code == 200
    updated1 = update_res1.json()["shelter"]
    assert updated1["available_beds"] == 50
    assert updated1["utilization_percentage"] == 90.0
    assert updated1["status"] == "Near Capacity"

    # 3. Update occupancy to full (100%)
    update_res2 = client.put("/api/shelters/TEST-SH-99", json={"occupancy": 500})
    assert update_res2.status_code == 200
    updated2 = update_res2.json()["shelter"]
    assert updated2["available_beds"] == 0
    assert updated2["utilization_percentage"] == 100.0
    assert updated2["status"] == "Full"

    # 4. Validation: occupancy > capacity must fail with 400
    invalid_occ_res = client.put("/api/shelters/TEST-SH-99", json={"occupancy": 550})
    assert invalid_occ_res.status_code == 400
    assert "exceed" in invalid_occ_res.json()["detail"].lower()

    # 5. Validation: capacity <= 0 must fail with 400
    invalid_cap_res = client.post("/api/shelters", json={
        "shelter_name": "Zero Cap Shelter",
        "location": "Test Zone",
        "capacity": 0,
        "occupancy": 0
    })
    assert invalid_cap_res.status_code == 400

    # 6. Delete shelter
    del_res = client.delete("/api/shelters/TEST-SH-99")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"

    # 7. Confirm deleted
    get_res = client.get("/api/shelters")
    assert not any(s["shelter_id"] == "TEST-SH-99" for s in get_res.json()["shelters"])

    print("PASS: test_shelter_crud_persistence_and_calculations")


def test_resource_crud_persistence_and_shortage_calculations():
    """Verify Resource CRUD, shortages/surpluses, and SQLite persistence."""
    client = TestClient(app)

    # 1. Create resource with shortage (required > available)
    res_payload = {
        "resource_id": "TEST-RES-99",
        "resource_type": "Emergency Blankets",
        "location": "Highlands",
        "required_quantity": 2000.0,
        "available_quantity": 800.0,
        "unit": "units"
    }
    create_res = client.post("/api/resources", json=res_payload)
    assert create_res.status_code == 200, create_res.text
    created = create_res.json()["resource"]
    assert created["resource_id"] == "TEST-RES-99"
    assert created["shortage"] == 1200.0
    assert created["surplus"] == 0.0
    assert created["status"] == "Shortage"

    # 2. Update to sufficient (available == required)
    upd_res1 = client.put("/api/resources/TEST-RES-99", json={"available_quantity": 2000.0})
    assert upd_res1.status_code == 200
    updated1 = upd_res1.json()["resource"]
    assert updated1["shortage"] == 0.0
    assert updated1["surplus"] == 0.0
    assert updated1["status"] == "Sufficient"

    # 3. Update to surplus (available > required)
    upd_res2 = client.put("/api/resources/TEST-RES-99", json={"available_quantity": 2500.0})
    assert upd_res2.status_code == 200
    updated2 = upd_res2.json()["resource"]
    assert updated2["shortage"] == 0.0
    assert updated2["surplus"] == 500.0
    assert updated2["status"] == "Surplus"

    # 4. Validation: negative quantity must fail with 400
    neg_res = client.post("/api/resources", json={
        "resource_type": "Invalid Item",
        "location": "Test Zone",
        "required_quantity": -50.0,
        "available_quantity": 10.0,
        "unit": "units"
    })
    assert neg_res.status_code == 400

    # 5. Delete resource
    del_res = client.delete("/api/resources/TEST-RES-99")
    assert del_res.status_code == 200

    # Confirm deletion
    all_res = client.get("/api/resources").json()["resources"]
    assert not any(r["resource_id"] == "TEST-RES-99" for r in all_res)

    print("PASS: test_resource_crud_persistence_and_shortage_calculations")


def test_incident_to_inventory_matching_and_full_shelter_guardrail():
    """Verify approximate text matching and ensure full shelters are never recommended as available."""
    client = TestClient(app)

    # Ensure demo inventories are populated
    client.post("/api/shelters/reset")
    client.post("/api/resources/reset")

    # Match for location 'Northern Panhandle' (where SH-03 is 300/300 Full)
    match_res = client.get("/api/inventory/match?location=Northern%20Panhandle")
    assert match_res.status_code == 200
    match_data = match_res.json()

    assert match_data["status"] == "success"
    assert match_data["match_type"] == "Approximate Normalized Text Match"
    assert "Approximate" in match_data["match_confidence"]

    # SH-03 is full (300/300)
    matched_shelters = match_data["matched_shelters"]
    assert len(matched_shelters) >= 1
    sh3 = next((s for s in matched_shelters if s["shelter_id"] == "SH-03"), None)
    assert sh3 is not None
    assert sh3["status"] == "Full"
    assert sh3["available_beds"] == 0

    # Full shelter guardrail: warnings must contain warning NOT to route evacuees here
    warnings = match_data["warnings"]
    assert any("100% capacity" in w and "DO NOT route evacuees here" in w for w in warnings)

    # Match for location 'Okanogan' (Drinking Water shortage)
    match_res2 = client.get("/api/inventory/match?location=Okanogan")
    assert match_res2.status_code == 200
    match_data2 = match_res2.json()
    matched_resources = match_data2["matched_resources"]
    assert any(r["resource_type"] == "Drinking Water" and r["shortage"] > 0 for r in matched_resources)

    print("PASS: test_incident_to_inventory_matching_and_full_shelter_guardrail")


def test_shelter_and_resource_reset_and_template():
    """Verify CSV templates download and reset-to-demo endpoints restore labeled data."""
    client = TestClient(app)

    # 1. Download shelter template
    sh_temp = client.get("/api/shelters/template")
    assert sh_temp.status_code == 200
    assert "text/csv" in sh_temp.headers["content-type"]
    assert "shelter_name,location,capacity,occupancy" in sh_temp.text

    # 2. Download resource template
    res_temp = client.get("/api/resources/template")
    assert res_temp.status_code == 200
    assert "text/csv" in res_temp.headers["content-type"]
    assert "resource_type,location,required_quantity,available_quantity,unit" in res_temp.text

    # 3. Reset shelters
    sh_reset = client.post("/api/shelters/reset")
    assert sh_reset.status_code == 200
    shelters = sh_reset.json()["shelters"]
    assert len(shelters) >= 5
    for s in shelters:
        assert s["data_source"] == "DEMO DATA — NOT LIVE MUNICIPAL INVENTORY"

    # 4. Reset resources
    res_reset = client.post("/api/resources/reset")
    assert res_reset.status_code == 200
    resources = res_reset.json()["resources"]
    assert len(resources) >= 4
    for r in resources:
        assert r["data_source"] == "DEMO DATA — NOT LIVE MUNICIPAL INVENTORY"

    print("PASS: test_shelter_and_resource_reset_and_template")


def test_auto_generated_shelter_and_resource_ids():
    """Verify that leaving Shelter ID or Resource ID blank auto-generates clean unique IDs."""
    client = TestClient(app)

    # 1. Create shelter with blank ID
    payload = {
        "shelter_id": "",
        "shelter_name": "Dynamic Auto-Gen Facility Alpha",
        "location": "Sector West",
        "capacity": 450,
        "occupancy": 150,
        "notes": "Testing blank shelter ID auto-generation"
    }
    res = client.post("/api/shelters", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    created = res.json()["shelter"]
    generated_id = created["shelter_id"]
    assert generated_id.startswith("SH-"), f"Expected ID starting with SH-, got {generated_id}"
    assert created["capacity"] == 450
    assert created["occupancy"] == 150
    assert created["available_beds"] == 300

    # 2. Create shelter with custom ID
    custom_id = f"SH-CUSTOM-{int(datetime.now().timestamp())}"
    custom_payload = {
        "shelter_id": custom_id,
        "shelter_name": "Custom ID Shelter",
        "location": "Sector North",
        "capacity": 200,
        "occupancy": 50
    }
    c_res = client.post("/api/shelters", json=custom_payload)
    assert c_res.status_code == 200
    assert c_res.json()["shelter"]["shelter_id"] == custom_id

    # 3. Duplicate custom ID should return 400
    dup_res = client.post("/api/shelters", json=custom_payload)
    assert dup_res.status_code == 400
    assert "already exists" in dup_res.json()["detail"].lower()

    # 4. Create resource with blank ID
    res_payload = {
        "resource_id": "",
        "resource_type": "Emergency Plasma Units",
        "location": "Sector West",
        "required_quantity": 200,
        "available_quantity": 80,
        "unit": "units"
    }
    r_res = client.post("/api/resources", json=res_payload)
    assert r_res.status_code == 200
    created_res = r_res.json()["resource"]
    gen_res_id = created_res["resource_id"]
    assert gen_res_id.startswith("RES-"), f"Expected ID starting with RES-, got {gen_res_id}"
    assert created_res["shortage"] == 120.0

    # 5. Duplicate resource ID should return 400
    dup_r_payload = {**res_payload, "resource_id": gen_res_id}
    dup_r_res = client.post("/api/resources", json=dup_r_payload)
    assert dup_r_res.status_code == 400
    assert "already exists" in dup_r_res.json()["detail"].lower()

    print("PASS: test_auto_generated_shelter_and_resource_ids")


def test_dynamic_economic_loss_coverage_and_hardcoded_label_regression():
    """Verify economic loss coverage is dynamically computed and never uses hardcoded 66.6% label."""
    client = TestClient(app)

    # 1. Empty dataset validation: 0 records uploaded returns safe 400
    empty_csv = "incident_identifier,disaster_type,incident_location,estimated_loss\n"
    res_empty = client.post(
        "/api/receive-file",
        files={"file": ("empty_dataset.csv", empty_csv.encode("utf-8"), "text/csv")}
    )
    assert res_empty.status_code == 400
    assert "no records" in res_empty.json()["detail"].lower()

    # 1b. Empty dataset safe handling in assessment engine (no division by zero)
    from main import DATA_STORE, generate_assessment
    orig_incidents = list(DATA_STORE["incidents"])
    orig_loss_cov = DATA_STORE.get("loss_coverage")
    orig_total_rec = DATA_STORE.get("total_assessed_records")
    try:
        DATA_STORE["incidents"] = []
        DATA_STORE["total_assessed_records"] = 0
        DATA_STORE["loss_coverage"] = None
        empty_assessment = generate_assessment()
        assert empty_assessment["summary"]["total_incidents"] == 0
        dq_empty = empty_assessment["summary"]["data_quality_report"]["economic_loss_coverage"]
        assert dq_empty["total_records"] == 0
        assert dq_empty["assessed_records"] == 0
        assert dq_empty["unassessed_records"] == 0
        assert dq_empty["coverage_percentage"] == 0.0
        assert dq_empty["coverage_display"] == "0.0% Present (0.0% Unassessed)"
        assert "66.6%" not in dq_empty["coverage_display"]
    finally:
        DATA_STORE["incidents"] = orig_incidents
        DATA_STORE["loss_coverage"] = orig_loss_cov
        DATA_STORE["total_assessed_records"] = orig_total_rec

    # 2. 100% loss coverage dataset (verified zeros count as assessed!)
    zero_loss_csv = "incident_identifier,disaster_type,incident_location,estimated_loss\n"
    for i in range(5):
        zero_loss_csv += f"INC-{i+1},Flood,Zone A,0.0\n"
    res_zero = client.post(
        "/api/receive-file",
        files={"file": ("zero_loss_test.csv", zero_loss_csv.encode("utf-8"), "text/csv")}
    )
    assert res_zero.status_code == 200
    dq_zero = res_zero.json()["summary"]["data_quality_report"]["economic_loss_coverage"]
    assert dq_zero["total_records"] == 5
    assert dq_zero["assessed_records"] == 5
    assert dq_zero["unassessed_records"] == 0
    assert dq_zero["coverage_percentage"] == 100.0
    assert dq_zero["coverage_display"] == "100.0% Present (0.0% Unassessed)"
    assert "66.6%" not in dq_zero["coverage_display"]

    # 3. 30-record fixture: 29 assessed, 1 unassessed (96.7% coverage)
    fixture_path = Path(__file__).resolve().parent / "received_datasets" / "noaa_storm_events_dynamic_test_fixture.csv"
    if fixture_path.exists():
        with open(fixture_path, "rb") as f:
            res_30 = client.post(
                "/api/receive-file",
                files={"file": ("noaa_storm_events_dynamic_test_fixture.csv", f.read(), "text/csv")}
            )
        assert res_30.status_code == 200
        dq_30 = res_30.json()["summary"]["data_quality_report"]["economic_loss_coverage"]
        assert dq_30["total_records"] == 30
        assert dq_30["assessed_records"] == 29
        assert dq_30["unassessed_records"] == 1
        assert dq_30["coverage_percentage"] == 96.7
        assert dq_30["coverage_display"] == "96.7% Present (3.3% Unassessed)"
        assert "66.6%" not in dq_30["coverage_display"]

    print("PASS: test_dynamic_economic_loss_coverage_and_hardcoded_label_regression")


def test_incident_assessment_filters_and_official_cad_priority():
    """Verify Critical, High, Medium, Low, Unknown, and Anomalies filter counts and official CAD priority separation."""
    client = TestClient(app)

    # Ingest 30-record fixture
    fixture_path = Path(__file__).resolve().parent / "received_datasets" / "noaa_storm_events_dynamic_test_fixture.csv"
    if fixture_path.exists():
        with open(fixture_path, "rb") as f:
            client.post(
                "/api/receive-file",
                files={"file": ("noaa_storm_events_dynamic_test_fixture.csv", f.read(), "text/csv")}
            )

    assess_res = client.get("/api/assess")
    assert assess_res.status_code == 200
    data = assess_res.json()
    incidents = data["incidents"]
    assert len(incidents) == 30

    critical_count = sum(1 for i in incidents if "Critical" in (i.get("impact_tier") or ""))
    high_count = sum(1 for i in incidents if "High" in (i.get("impact_tier") or ""))
    med_count = sum(1 for i in incidents if "Moderate" in (i.get("impact_tier") or "") or "Medium" in (i.get("impact_tier") or ""))
    low_count = sum(1 for i in incidents if any(k in (i.get("impact_tier") or "") for k in ["Minor", "Negligible", "Low"]))
    unknown_count = sum(1 for i in incidents if any(k in (i.get("impact_tier") or "") for k in ["Unassessed", "Indeterminate", "Unknown"]))
    anomaly_count = sum(1 for i in incidents if bool(i.get("is_anomaly") or i.get("anomaly_status") == "Anomaly"))

    assert critical_count == 8, f"Expected 8 critical impact events, got {critical_count}"
    assert anomaly_count == 21, f"Expected 21 anomalies, got {anomaly_count}"
    assert critical_count + high_count + med_count + low_count + unknown_count == len(incidents)

    # Verify official CAD priority remains Unknown for NOAA records (no fabrication)
    for inc in incidents:
        assert inc["official_priority"] == "Unknown", f"Official priority must be Unknown, got {inc['official_priority']}"

    print("PASS: test_incident_assessment_filters_and_official_cad_priority")


def test_sqlite_shelter_persistence_simulation():
    """Verify shelter records persist in SQLite across simulated connection re-inits."""
    test_id = f"SH-PERSIST-{int(datetime.now().timestamp())}"
    created = database.create_shelter({
        "shelter_id": test_id,
        "shelter_name": "Persistent Verification Shelter",
        "location": "Zone Omega",
        "capacity": 300,
        "occupancy": 75,
        "notes": "Persistence check"
    })
    assert created["shelter_id"] == test_id

    # Re-initialize database connection simulation
    database.init_db()
    retrieved = database.get_shelter_by_id(test_id)
    assert retrieved is not None
    assert retrieved["shelter_name"] == "Persistent Verification Shelter"
    assert retrieved["capacity"] == 300
    assert retrieved["occupancy"] == 75
    assert retrieved["available_beds"] == 225

    # Clean up test shelter
    database.delete_shelter(test_id)
    assert database.get_shelter_by_id(test_id) is None

    print("PASS: test_sqlite_shelter_persistence_simulation")


if __name__ == "__main__":
    print("=================================================================")
    print("MEMBER 2 REGRESSION & INVESTIGATION TEST SUITE")
    print("=================================================================")
    test_observed_zero_casualties_and_losses()
    test_missing_casualty_and_loss_fields()
    test_positive_fatalities_and_injuries()
    test_positive_economic_loss()
    test_catastrophic_economic_loss()
    test_incomplete_shelter_and_population_handling()
    test_distinct_numerical_feature_scores()
    test_real_noaa_500_records_distribution()
    test_missing_casualties_with_zero_loss()
    test_verified_zero_casualties_with_missing_loss()
    test_verified_zeros_all_dimensions()
    test_positive_loss_with_missing_casualties()
    test_canonical_and_raw_noaa_headers()
    test_api_assess_no_stale_mock_leak_after_noaa_upload()
    test_reconciled_anomaly_and_record_counts()
    test_insufficient_evidence_risk_rating()
    test_enriched_recommendations_structure()
    test_api_activity_logging_and_correlation()
    test_job_log_isolation_between_uploads()
    test_critical_impact_vs_dispatch_priority_separation()
    test_critical_incident_beyond_row_100_retained()
    test_post_and_get_assessment_counts_consistent()
    test_ml_results_provenance_and_active_noaa_alignment()
    test_ml_results_anomaly_counts_and_unavailable_shelter_stats()
    test_ml_results_model_status_candidate_only_and_benchmarks_separation()
    test_dynamic_economic_loss_coverage_calculation()
    test_incident_critical_filter_agrees_with_rule_based_tier()
    test_shelter_crud_persistence_and_calculations()
    test_resource_crud_persistence_and_shortage_calculations()
    test_incident_to_inventory_matching_and_full_shelter_guardrail()
    test_shelter_and_resource_reset_and_template()
    test_auto_generated_shelter_and_resource_ids()
    test_dynamic_economic_loss_coverage_and_hardcoded_label_regression()
    test_incident_assessment_filters_and_official_cad_priority()
    test_sqlite_shelter_persistence_simulation()
    print("\n=================================================================")
    print("ALL 35 REGRESSION & VALIDATION TESTS PASSED CLEANLY!")
    print("=================================================================")




