"""Integration test suite for Member 2 Disaster Assessment API.

Verifies:
1. Health check (GET /api/health)
2. File upload & real ML inference (POST /api/receive-file) with 20 canonical columns
3. Handling of missing/null values (null priority, null affected population, etc.)
4. Error responses for empty files and unsupported formats
5. Dashboard compatibility endpoints (GET /api/assess, GET /api/processed-results)
6. Validation that Isolation Forest predictions and decision scores are genuine, not mocked
"""
import io
import json
from fastapi.testclient import TestClient

from main import app, load_isolation_forest_artifact

client = TestClient(app)

SAMPLE_20_COLUMN_CSV = """incident_identifier,disaster_type,incident_location,priority_level,affected_population,reported_casualties,incident_timestamp,response_status,latitude,longitude,incident_description,injured_population,displaced_population,required_resources,available_resources,available_shelter_capacity,current_shelter_occupancy,responding_agency,infrastructure_damage,estimated_loss
EVT-001,Flood,Nellore,Critical,12000,18,2026-10-18 10:30,Responding,14.4426,79.9865,Severe flooding along coastal districts,140,3200,Boats,Rescue Teams,5000,4800,NDRF,Severe,85000000
EVT-002,Cyclone,Visakhapatnam,High,4500,3,2026-10-18 11:15,Assessing,17.6868,83.2185,Tropical cyclone approach and high winds,25,800,Medical Kits,Food Packs,3000,1200,SDRF,Moderate,25000000
EVT-003,Earthquake,Chittoor,Medium,1800,0,2026-10-18 12:00,Contained,13.2172,79.1003,Moderate tremor felt across municipal zone,5,120,Water,Blankets,2000,400,Local Police,Low,5000000
EVT-004,Landslide,Araku,Low,350,1,2026-10-18 12:45,Contained,18.3273,82.8775,Minor slope failure blocking rural highway,2,40,Excavators,Rescue Teams,1000,150,Municipal,Low,1200000
"""

SAMPLE_CSV_WITH_NULLS = (
    "incident_identifier,disaster_type,incident_location,priority_level,affected_population,"
    "reported_casualties,incident_timestamp,response_status,latitude,longitude,"
    "incident_description,injured_population,displaced_population,required_resources,"
    "available_resources,available_shelter_capacity,current_shelter_occupancy,"
    "responding_agency,infrastructure_damage,estimated_loss\n"
    "EVT-101,Flood,Kadapa,,9500,,2026-10-19 08:00,Reported,14.4673,78.8242,"
    "Flooding with unverified casualty reports,,1500,Boats,Food,4000,3800,NDRF,Moderate,250000\n"
    "EVT-102,Forest Fire,Tirupati,High,,,2026-10-19 09:30,Assessing,13.6288,79.4192,"
    "Scattered fire reports near foothills,,50,Fire Engines,Water,,,Police,Low,\n"
)


def test_health_check():
    """Verify backend health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "last_updated" in data


def test_model_artifact_loading():
    """Verify that the real Isolation Forest artifact exists and can be loaded."""
    artifact = load_isolation_forest_artifact()
    assert artifact is not None, "Isolation Forest artifact could not be loaded"
    assert "model" in artifact
    assert "imputer" in artifact
    assert "scaler" in artifact
    assert "feature_columns" in artifact
    expected_cols = [
        "affected_population", "reported_casualties", "injured_population",
        "displaced_population", "available_shelter_capacity",
        "current_shelter_occupancy", "estimated_loss", "latitude", "longitude"
    ]
    assert artifact["feature_columns"] == expected_cols


def test_receive_file_with_valid_csv():
    """Upload valid 20-canonical-column CSV and verify real ML inference."""
    file_bytes = io.BytesIO(SAMPLE_20_COLUMN_CSV.encode("utf-8"))
    response = client.post(
        "/api/receive-file",
        files={"file": ("test_member1_data.csv", file_bytes, "text/csv")}
    )
    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()

    # Structural verification
    assert data["status"] == "success"
    assert data["total_records"] == 4
    assert data["valid_records_count"] == 4
    assert data["insufficient_data_records_count"] == 0

    # Summary verification
    summary = data["summary"]
    assert summary["total_incidents"] == 4
    assert summary["critical_incidents"] == 1
    assert summary["high_priority_incidents"] == 1
    assert summary["medium_priority_incidents"] == 1
    assert summary["low_priority_incidents"] == 1
    assert summary["overall_risk"] in ("HIGH", "MEDIUM", "LOW")

    # Real ML assessment verification
    ml = data["ml_assessment"]
    assert "Isolation Forest" in ml["model_name"]
    assert ml["model_status"].startswith("Operational")
    assert isinstance(ml["anomalies_flagged"], int)
    assert 0.0 <= ml["anomaly_fraction"] <= 1.0

    # Verify evaluated sample records have real decision scores (floats)
    records = data["sample_evaluated_records"]
    assert len(records) == 4
    for r in records:
        assert "anomaly_score" in r
        assert isinstance(r["anomaly_score"], float)
        assert "is_anomaly" in r
        assert isinstance(r["is_anomaly"], bool)
        assert r["priority_level"] in ("Critical", "High", "Medium", "Low")

    # Verify recommendations are present and computed
    assert len(data["recommendations"]) > 0


def test_receive_file_with_missing_values():
    """Upload CSV with null priority_level and missing numeric fields.
    Verifies that null priority becomes 'Unknown' and imputer handles numeric gaps."""
    file_bytes = io.BytesIO(SAMPLE_CSV_WITH_NULLS.encode("utf-8"))
    response = client.post(
        "/api/receive-file",
        files={"file": ("test_missing_data.csv", file_bytes, "text/csv")}
    )
    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()

    assert data["status"] == "success"
    assert data["total_records"] == 2

    summary = data["summary"]
    # Row 1 had null priority_level, should be categorized as unknown
    assert summary["unknown_priority_incidents"] >= 1
    # Row 2 had High priority
    assert summary["high_priority_incidents"] >= 1

    # Verify missing cells were tracked
    ml = data["ml_assessment"]
    assert ml["missing_feature_cells"] > 0
    assert "missing_cells_per_feature" in ml

    # Verify sample records reflect 'Unknown' for null priority
    records = data["sample_evaluated_records"]
    priorities = [r["priority_level"] for r in records]
    assert "Unknown" in priorities


def test_receive_file_empty_error():
    """Verify 400 Bad Request when an empty file is uploaded."""
    empty_bytes = io.BytesIO(b"")
    response = client.post(
        "/api/receive-file",
        files={"file": ("empty.csv", empty_bytes, "text/csv")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_receive_file_unsupported_extension_error():
    """Verify 415 error when unsupported file type is uploaded."""
    txt_bytes = io.BytesIO(b"some text content")
    response = client.post(
        "/api/receive-file",
        files={"file": ("test.txt", txt_bytes, "text/plain")}
    )
    assert response.status_code == 415


def test_dashboard_endpoints_reflect_uploaded_data():
    """Verify that after receiving a CSV, GET /api/assess and GET /api/processed-results work."""
    # First upload
    file_bytes = io.BytesIO(SAMPLE_20_COLUMN_CSV.encode("utf-8"))
    client.post(
        "/api/receive-file",
        files={"file": ("integration_test_processed.csv", file_bytes, "text/csv")}
    )

    # Check /api/assess
    assess_res = client.get("/api/assess")
    assert assess_res.status_code == 200
    assess_data = assess_res.json()
    assert "summary" in assess_data
    assert "incidents" in assess_data
    assert "shelters" in assess_data

    # Check /api/processed-results
    proc_res = client.get("/api/processed-results")
    assert proc_res.status_code == 200
    proc_data = proc_res.json()
    assert proc_data["status"] == "success"
    assert proc_data["total_records"] >= 4
    assert len(proc_data["records"]) >= 4


if __name__ == "__main__":
    print("Running integration tests...")
    test_health_check()
    print("PASS: test_health_check")
    test_model_artifact_loading()
    print("PASS: test_model_artifact_loading")
    test_receive_file_with_valid_csv()
    print("PASS: test_receive_file_with_valid_csv")
    test_receive_file_with_missing_values()
    print("PASS: test_receive_file_with_missing_values")
    test_receive_file_empty_error()
    print("PASS: test_receive_file_empty_error")
    test_receive_file_unsupported_extension_error()
    print("PASS: test_receive_file_unsupported_extension_error")
    test_dashboard_endpoints_reflect_uploaded_data()
    print("PASS: test_dashboard_endpoints_reflect_uploaded_data")
    print("\nALL 7 TESTS PASSED SUCCESSFULLY!")
