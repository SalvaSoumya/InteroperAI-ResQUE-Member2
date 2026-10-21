from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
import io
import json
import math

from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Query, Response
from fastapi.middleware.cors import CORSMiddleware
import uuid

from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import joblib

try:
    import database
except ImportError:
    from . import database

# Ensure SQLite database tables and demo seed records are initialized
database.init_db()


# ============================================================
# ML ARTIFACT LOADING
# ============================================================

ML_ARTIFACT_DATA = None
HYBRID_ARTIFACT_DATA = None

def get_ml_artifacts_dir() -> Path:
    p1 = Path(__file__).resolve().parent / "received_datasets" / "member2_ml_artifacts"
    if p1.exists() and (p1 / "isolation_forest_model.joblib").exists():
        return p1
    p2 = Path(__file__).resolve().parent.parent / "member2_ml_artifacts"
    if p2.exists() and (p2 / "isolation_forest_model.joblib").exists():
        return p2
    return p1


def get_hybrid_artifacts_dir() -> Path:
    p1 = Path(__file__).resolve().parent.parent / "member2_hybrid_artifacts"
    if p1.exists() and (p1 / "hybrid_isolation_forest.joblib").exists():
        return p1
    p2 = Path(__file__).resolve().parent / "received_datasets" / "member2_hybrid_artifacts"
    if p2.exists() and (p2 / "hybrid_isolation_forest.joblib").exists():
        return p2
    return p1


def load_hybrid_artifact():
    global HYBRID_ARTIFACT_DATA
    if HYBRID_ARTIFACT_DATA is not None:
        return HYBRID_ARTIFACT_DATA
    h_dir = get_hybrid_artifacts_dir()
    h_path = h_dir / "hybrid_isolation_forest.joblib"
    if not h_path.exists():
        return None
    try:
        loaded = joblib.load(h_path)
        HYBRID_ARTIFACT_DATA = loaded
        return HYBRID_ARTIFACT_DATA
    except Exception as exc:
        print(f"Warning: Could not load hybrid artifact from {h_path}: {exc}")
        return None


def load_isolation_forest_artifact():
    global ML_ARTIFACT_DATA
    if ML_ARTIFACT_DATA is not None:
        return ML_ARTIFACT_DATA
    artifacts_dir = get_ml_artifacts_dir()
    model_path = artifacts_dir / "isolation_forest_model.joblib"
    if not model_path.exists():
        print(f"Warning: Isolation Forest artifact not found at {model_path}")
        return None
    try:
        loaded = joblib.load(model_path)
        ML_ARTIFACT_DATA = loaded
        return ML_ARTIFACT_DATA
    except Exception as exc:
        print(f"Error loading Isolation Forest artifact from {model_path}: {exc}")
        return None


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Intelligent Disaster Assessment and Resource Management API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INITIAL DATA
# ============================================================
# These are starting records only.
# The /api/receive endpoint can update them while the
# application is running.

INITIAL_INCIDENTS = [
    {
        "id": "INC-001",
        "zone": "Zone A",
        "type": "Flood",
        "severity": 5,
        "affected": 4200,
        "vulnerability": 5,
        "distance": 2.5
    },
    {
        "id": "INC-002",
        "zone": "Zone B",
        "type": "Cyclone",
        "severity": 3,
        "affected": 1500,
        "vulnerability": 3,
        "distance": 7.0
    },
    {
        "id": "INC-003",
        "zone": "Zone C",
        "type": "Flood",
        "severity": 5,
        "affected": 6200,
        "vulnerability": 5,
        "distance": 1.8
    },
    {
        "id": "INC-004",
        "zone": "Zone D",
        "type": "Landslide",
        "severity": 4,
        "affected": 2100,
        "vulnerability": 4,
        "distance": 4.2
    }
]


INITIAL_SHELTERS = [
    {
        "id": "SH-01",
        "name": "Community Hall A",
        "zone": "Zone A",
        "capacity": 1000,
        "occupancy": 920
    },
    {
        "id": "SH-02",
        "name": "School B",
        "zone": "Zone B",
        "capacity": 1200,
        "occupancy": 540
    },
    {
        "id": "SH-03",
        "name": "Relief Centre C",
        "zone": "Zone C",
        "capacity": 800,
        "occupancy": 760
    },
    {
        "id": "SH-04",
        "name": "Sports Complex D",
        "zone": "Zone D",
        "capacity": 1500,
        "occupancy": 610
    }
]


INITIAL_RESOURCES = [
    {
        "zone": "Zone A",
        "type": "Water",
        "required": 5000,
        "available": 1800,
        "unit": "L"
    },
    {
        "zone": "Zone A",
        "type": "Medical Kits",
        "required": 500,
        "available": 180,
        "unit": "kits"
    },
    {
        "zone": "Zone B",
        "type": "Water",
        "required": 3000,
        "available": 3200,
        "unit": "L"
    },
    {
        "zone": "Zone C",
        "type": "Food Packs",
        "required": 4500,
        "available": 2900,
        "unit": "packs"
    },
    {
        "zone": "Zone C",
        "type": "Medical Kits",
        "required": 400,
        "available": 120,
        "unit": "kits"
    },
    {
        "zone": "Zone D",
        "type": "Blankets",
        "required": 1800,
        "available": 1500,
        "unit": "units"
    }
]


# ============================================================
# LIVE IN-MEMORY DATA STORE
# ============================================================

DATA_STORE = {
    "incidents": deepcopy(INITIAL_INCIDENTS),
    "shelters": database.get_all_shelters() or deepcopy(INITIAL_SHELTERS),
    "resources": database.get_all_resources() or deepcopy(INITIAL_RESOURCES),
    "anomalies": None,
    "recommendations": None,
    "last_updated": datetime.now().isoformat(),
    "current_job_id": None,
    "current_request_id": None,
    "current_correlation_id": None,
    "current_assessment_id": None,
    "current_client_id": None,
    "current_filename": None,
    "total_assessed_records": None,
    "anomalies_detected": None,
    "critical_impact_events": None,
    "dispatch_priority_critical_count": None,
    "overall_risk": None,
    "overall_risk_reason": None,
    "loss_coverage": None,
}

# In-memory API activity log ring buffer (isolated across requests)
API_ACTIVITY_LOGS = []
MAX_ACTIVITY_LOGS = 50

def record_api_activity(
    timestamp: str,
    client_id: str,
    filename: str,
    request_id: str,
    job_id: str,
    assessment_id: Optional[str],
    correlation_id: str,
    records_count: int,
    processing_status: str,
    http_status: int,
    error_message: Optional[str] = None
) -> dict:
    entry = {
        "timestamp": timestamp,
        "client_id": client_id,
        "filename": filename,
        "request_id": request_id,
        "job_id": job_id,
        "assessment_id": assessment_id or "N/A",
        "correlation_id": correlation_id,
        "records_count": records_count,
        "processing_status": processing_status,
        "http_status": http_status,
        "error_message": error_message
    }
    API_ACTIVITY_LOGS.insert(0, entry)
    while len(API_ACTIVITY_LOGS) > MAX_ACTIVITY_LOGS:
        API_ACTIVITY_LOGS.pop()
    return entry


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def update_timestamp():
    DATA_STORE["last_updated"] = datetime.now().isoformat()


def safe_number(value: Any, default: float = 0) -> float:
    """
    Convert a value to a number safely, returning default on NaN/Inf or conversion errors.
    """
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return default
        return val
    except (TypeError, ValueError):
        return default


def parse_damage_string(val: Any) -> float:
    """Parse NOAA damage strings (e.g., '10.00K', '2.50M', '1.00B', '0.00K', 0)."""
    if pd.isna(val) or str(val).strip() in ("", "nan", "none", "null", "<NA>"):
        return np.nan
    s = str(val).strip().upper()
    if s in ("0", "0.00", "0.00K", "0.00M", "0.00B"):
        return 0.0
    mult = 1.0
    if s.endswith("K"):
        mult = 1e3
        s = s[:-1]
    elif s.endswith("M"):
        mult = 1e6
        s = s[:-1]
    elif s.endswith("B"):
        mult = 1e9
        s = s[:-1]
    try:
        return float(s) * mult
    except (ValueError, TypeError):
        return np.nan


# ============================================================
# INCIDENT PRIORITY ENGINE
# ============================================================

def priority(incident: dict) -> float:
    """
    Explainable disaster priority score.

    Factors:
    - Severity
    - Affected population
    - Vulnerability
    - Distance/proximity
    """

    severity = safe_number(incident.get("severity"))
    affected = safe_number(incident.get("affected"))
    vulnerability = safe_number(incident.get("vulnerability"))
    distance = safe_number(incident.get("distance"))

    score = (
        severity * 10
        + min(affected / 1000, 10) * 4
        + vulnerability * 4
        + max(0, 10 - distance) * 2
    )

    return round(min(score, 100), 1)


def assess_incidents():
    results = []

    for incident in DATA_STORE["incidents"]:
        # If incident already has priority_score and priority (from CSV assessment), preserve them
        if "priority_score" in incident and "priority" in incident:
            results.append(incident)
            continue

        score = priority(incident)

        if score >= 80:
            level = "Critical"
        elif score >= 60:
            level = "High"
        else:
            level = "Medium"

        results.append(
            {
                **incident,
                "priority_score": score,
                "priority": level
            }
        )

    return sorted(
        results,
        key=lambda item: item.get("priority_score", 0),
        reverse=True
    )


# ============================================================
# SHELTER ASSESSMENT ENGINE
# ============================================================

def assess_shelters():
    results = []
    for shelter in DATA_STORE.get("shelters", []):
        capacity = safe_number(shelter.get("capacity"))
        occupancy = safe_number(shelter.get("occupancy"))
        utilization = round(occupancy / capacity * 100, 1) if capacity > 0 else 0.0
        if utilization >= 100.0:
            status = "Full"
        elif utilization >= 80.0:
            status = "Near Capacity"
        else:
            status = "Available"

        results.append(
            {
                **shelter,
                "utilization": utilization,
                "utilization_percentage": utilization,
                "available_beds": max(int(capacity - occupancy), 0),
                "available": max(int(capacity - occupancy), 0),
                "status": status
            }
        )
    return results


# ============================================================
# RESOURCE ASSESSMENT ENGINE
# ============================================================

def assess_resources():
    results = []
    for resource in DATA_STORE.get("resources", []):
        req = safe_number(resource.get("required_quantity", resource.get("required", 0)))
        avail = safe_number(resource.get("available_quantity", resource.get("available", 0)))
        shortage = max(req - avail, 0)
        surplus = max(avail - req, 0)
        avail_pct = round(avail / req * 100, 1) if req > 0 else 100.0

        if shortage > 0:
            status = "Critical Shortage" if avail_pct < 50.0 else "Shortage"
        elif surplus > 0:
            status = "Surplus"
        else:
            status = "Sufficient"

        results.append(
            {
                **resource,
                "required_quantity": req,
                "available_quantity": avail,
                "shortage": shortage,
                "surplus": surplus,
                "availability_percent": avail_pct,
                "status": status
            }
        )
    return results


# ============================================================
# ANOMALY DETECTION
# ============================================================

# ============================================================
# ANOMALY DETECTION
# ============================================================

def anomalies(shelters):
    """
    Detect unusual shelter conditions using Isolation Forest.

    Features used:
    - occupancy
    - capacity
    - utilization percentage
    """

    if not shelters:
        return []

    features = [
        [
            safe_number(x.get("occupancy")),
            safe_number(x.get("capacity")),
            safe_number(x.get("utilization"))
        ]
        for x in shelters
    ]

    model = IsolationForest(
        n_estimators=100,
        contamination=0.25,
        random_state=42
    )

    predictions = model.fit_predict(features)
    scores = model.decision_function(features)

    results = []

    for shelter, prediction, score in zip(
        shelters,
        predictions,
        scores
    ):
        results.append(
            {
                "shelter": shelter["name"],
                "zone": shelter["zone"],
                "utilization": shelter["utilization"],
                "anomaly": bool(prediction == -1),
                "anomaly_score": round(float(score), 4),
                "status": (
                    "Unusual"
                    if prediction == -1
                    else "Normal"
                )
            }
        )

    return results

# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def recommendations(
    incidents,
    shelters,
    resources,
    anomaly_results
):

    recommendations_list = []

    # --------------------------------------------------------
    # INCIDENT RECOMMENDATIONS
    # --------------------------------------------------------

    for incident in incidents:

        if incident["priority"] in (
            "Critical",
            "High"
        ):

            recommendations_list.append(
                {
                    "priority": incident["priority"],
                    "zone": incident["zone"],
                    "action": (
                        f"Deploy response team to "
                        f"{incident['zone']} and prioritize "
                        f"{incident['type']} incident."
                    )
                }
            )

    # --------------------------------------------------------
    # RESOURCE RECOMMENDATIONS
    # --------------------------------------------------------

    for resource in resources:

        if resource["status"] != "Sufficient":

            priority_level = (
                "Critical"
                if resource["status"] == "Critical Shortage"
                else "High"
            )

            recommendations_list.append(
                {
                    "priority": priority_level,
                    "zone": resource["zone"],
                    "action": (
                        f"Reallocate or dispatch "
                        f"{resource['shortage']} "
                        f"{resource['unit']} of "
                        f"{resource['type']}."
                    )
                }
            )

    # --------------------------------------------------------
    # SHELTER RECOMMENDATIONS
    # --------------------------------------------------------

    for shelter in shelters:

        if shelter["status"] != "Available":

            priority_level = (
                "Critical"
                if shelter["status"] == "Critical"
                else "Medium"
            )

            recommendations_list.append(
                {
                    "priority": priority_level,
                    "zone": shelter["zone"],
                    "action": (
                        f"Monitor {shelter['name']} "
                        f"({shelter['utilization']}% occupied) "
                        f"and prepare alternate shelter capacity."
                    )
                }
            )

    # --------------------------------------------------------
    # ML ANOMALY RECOMMENDATIONS
    # --------------------------------------------------------

    for anomaly in anomaly_results:

        if anomaly["anomaly"]:

            recommendations_list.append(
                {
                    "priority": "High",
                    "zone": anomaly["zone"],
                    "action": (
                        f"Investigate unusual conditions at "
                        f"{anomaly['shelter']} "
                        f"({anomaly['utilization']}% utilization) "
                        f"identified by the Isolation Forest model."
                    )
                }
            )

    # --------------------------------------------------------
    # PRIORITY ORDER
    # --------------------------------------------------------

    priority_order = {
        "Critical": 1,
        "High": 2,
        "Medium": 3
    }

    recommendations_list.sort(
        key=lambda item: priority_order.get(
            item["priority"],
            99
        )
    )

    return recommendations_list


    

def generate_assessment():

    incidents = assess_incidents()

    shelters = assess_shelters()

    resources = assess_resources()

    anomaly_results = DATA_STORE.get("anomalies") if DATA_STORE.get("anomalies") is not None else anomalies(shelters)

    if DATA_STORE.get("recommendations") is not None:
        recommendation_results = list(DATA_STORE["recommendations"])
    else:
        recommendation_results = recommendations(
            incidents,
            shelters,
            resources,
            anomaly_results
        )

    if DATA_STORE.get("dispatch_priority_critical_count") is not None:
        dispatch_priority_critical = int(DATA_STORE["dispatch_priority_critical_count"])
        critical_incidents = dispatch_priority_critical
    else:
        critical_incidents = sum(
            item["priority"] == "Critical"
            for item in incidents
        )
        dispatch_priority_critical = critical_incidents

    if DATA_STORE.get("critical_impact_events") is not None:
        critical_impact_events = int(DATA_STORE["critical_impact_events"])
    else:
        critical_impact_events = sum(
            item.get("impact_tier") == "Critical Impact"
            for item in incidents
        ) if any(item.get("impact_tier") for item in incidents) else critical_incidents

    shelter_risks = sum(
        item["status"] != "Available"
        for item in shelters
    )

    resource_shortages = sum(
        item["status"] != "Sufficient"
        for item in resources
    )

    # Reconciled anomaly count from current assessment job if present, else sum from anomaly_results
    if DATA_STORE.get("anomalies_detected") is not None:
        anomaly_count = int(DATA_STORE["anomalies_detected"])
    else:
        anomaly_count = sum(
            bool(item.get("anomaly", False))
            for item in anomaly_results
        )

    total_records = DATA_STORE.get("total_assessed_records") if DATA_STORE.get("total_assessed_records") is not None else len(incidents)

    # Dynamic loss coverage calculation
    loss_cov = DATA_STORE.get("loss_coverage")
    if not loss_cov:
        total_i = len(incidents)
        def _is_valid_loss(val):
            if val is None:
                return False
            try:
                f = float(val)
                return not (math.isnan(f) or math.isinf(f))
            except (ValueError, TypeError):
                return False

        if total_i > 0 and any("estimated_loss" in inc for inc in incidents):
            assessed_l = sum(1 for inc in incidents if _is_valid_loss(inc.get("estimated_loss")))
            unassessed_l = total_i - assessed_l
            cov_pct = round(assessed_l / total_i * 100.0, 1) if total_i > 0 else 0.0
            unass_pct = round(unassessed_l / total_i * 100.0, 1) if total_i > 0 else 0.0
            loss_cov = {
                "total_records": total_i,
                "assessed_records": assessed_l,
                "unassessed_records": unassessed_l,
                "coverage_percentage": cov_pct,
                "unassessed_percentage": unass_pct,
                "coverage_display": f"{cov_pct}% Present ({unass_pct}% Unassessed)"
            }
        else:
            loss_cov = {
                "total_records": total_i,
                "assessed_records": 0,
                "unassessed_records": total_i,
                "coverage_percentage": 0.0,
                "unassessed_percentage": 100.0 if total_i > 0 else 0.0,
                "coverage_display": "0.0% Present (100.0% Unassessed)" if total_i > 0 else "0.0% Present (0.0% Unassessed)"
            }

    # Evidence-based Overall Risk: use assessed overall risk if available, else derive from initial incidents
    if DATA_STORE.get("overall_risk") is not None:
        overall_risk = DATA_STORE["overall_risk"]
        overall_risk_reason = DATA_STORE.get("overall_risk_reason", "")
    else:
        # Fallback baseline when no CSV has been assessed yet
        if critical_incidents >= 2:
            overall_risk = "HIGH"
            overall_risk_reason = f"{critical_incidents} critical priority incidents active in current baseline."
        elif critical_incidents >= 1:
            overall_risk = "MEDIUM"
            overall_risk_reason = f"{critical_incidents} critical priority incident active in current baseline."
        else:
            overall_risk = "LOW"
            overall_risk_reason = "No critical incidents in current baseline."

    shelter_metrics = database.get_shelter_summary_metrics()
    resource_metrics = database.get_resource_summary_metrics()

    return {
        "summary": {
            "total_incidents": total_records,
            "total_assessed_records": total_records,
            "filename": DATA_STORE.get("current_filename"),
            "critical_incidents": critical_incidents,
            "critical_impact_events": critical_impact_events,
            "dispatch_priority_critical_count": dispatch_priority_critical,
            "shelter_risks": shelter_risks,
            "resource_shortages": resource_shortages,
            "anomalies": anomaly_count,
            "anomalies_detected": anomaly_count,
            "overall_risk": overall_risk,
            "overall_risk_reason": overall_risk_reason,
            "loss_coverage": loss_cov,
            "data_quality_report": {
                "economic_loss_coverage": loss_cov
            },
            "shelter_summary": shelter_metrics,
            "resource_summary": resource_metrics,
            "last_updated": DATA_STORE["last_updated"],
            "job_id": DATA_STORE.get("current_job_id"),
            "request_id": DATA_STORE.get("current_request_id"),
            "correlation_id": DATA_STORE.get("current_correlation_id"),
            "assessment_id": DATA_STORE.get("current_assessment_id"),
            "client_id": DATA_STORE.get("current_client_id")
        },

        "incidents": incidents,

        "shelters": shelters,

        "resources": resources,

        "shelter_summary": shelter_metrics,

        "resource_summary": resource_metrics,

        "loss_coverage": loss_cov,

        "data_quality_report": {
            "economic_loss_coverage": loss_cov
        },

        "anomalies": anomaly_results,

        "recommendations": recommendation_results
    }


# ============================================================
# API ROUTES
# ============================================================

@app.get("/")
def root():

    return {
        "message": (
            "Intelligent Disaster Assessment and "
            "Resource Management API is running"
        )
    }


@app.get("/api/health")
def health():

    return {
        "status": "healthy",
        "last_updated": DATA_STORE["last_updated"]
    }


@app.get("/api/assess")
def assess():

    return generate_assessment()


# ============================================================
# RECEIVE NEW DATA
# ============================================================

@app.post("/api/receive")
def receive(payload: dict):

    updated = {
        "incidents": 0,
        "shelters": 0,
        "resources": 0
    }

    # --------------------------------------------------------
    # OPTION 1:
    # Receive complete lists
    #
    # {
    #   "incidents": [...],
    #   "shelters": [...],
    #   "resources": [...]
    # }
    # --------------------------------------------------------

    if "incidents" in payload:

        if not isinstance(
            payload["incidents"],
            list
        ):
            raise HTTPException(
                status_code=400,
                detail="'incidents' must be a list"
            )

        DATA_STORE["incidents"] = deepcopy(
            payload["incidents"]
        )

        updated["incidents"] = len(
            payload["incidents"]
        )

    if "shelters" in payload:

        if not isinstance(
            payload["shelters"],
            list
        ):
            raise HTTPException(
                status_code=400,
                detail="'shelters' must be a list"
            )

        DATA_STORE["shelters"] = deepcopy(
            payload["shelters"]
        )

        updated["shelters"] = len(
            payload["shelters"]
        )

    if "resources" in payload:

        if not isinstance(
            payload["resources"],
            list
        ):
            raise HTTPException(
                status_code=400,
                detail="'resources' must be a list"
            )

        DATA_STORE["resources"] = deepcopy(
            payload["resources"]
        )

        updated["resources"] = len(
            payload["resources"]
        )

    # --------------------------------------------------------
    # OPTION 2:
    # Receive records using a common "records" array.
    #
    # Example:
    #
    # {
    #   "records": [
    #       {
    #           "category": "incident",
    #           ...
    #       }
    #   ]
    # }
    # --------------------------------------------------------

    if "records" in payload:

        records = payload["records"]

        if not isinstance(records, list):

            raise HTTPException(
                status_code=400,
                detail="'records' must be a list"
            )

        for record in records:

            if not isinstance(record, dict):
                continue

            category = str(
                record.get(
                    "category",
                    record.get("type_category", "")
                )
            ).lower()

            clean_record = {
                key: value
                for key, value in record.items()
                if key not in (
                    "category",
                    "type_category"
                )
            }

            if category == "incident":

                DATA_STORE["incidents"].append(
                    clean_record
                )

                updated["incidents"] += 1

            elif category == "shelter":

                DATA_STORE["shelters"].append(
                    clean_record
                )

                updated["shelters"] += 1

            elif category == "resource":

                DATA_STORE["resources"].append(
                    clean_record
                )

                updated["resources"] += 1

    # --------------------------------------------------------
    # Nothing useful was supplied
    # --------------------------------------------------------

    if sum(updated.values()) == 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "No valid disaster records were received. "
                "Send incidents, shelters, resources, "
                "or records."
            )
        )

    update_timestamp()

    # Immediately generate the new assessment.
    assessment = generate_assessment()

    return {
        "status": "received",
        "updated": updated,
        "last_updated": DATA_STORE["last_updated"],
        "assessment": assessment
    }
MODEL_FEATURE_COLS = [
    "affected_population",
    "reported_casualties",
    "injured_population",
    "displaced_population",
    "available_shelter_capacity",
    "current_shelter_occupancy",
    "estimated_loss",
    "latitude",
    "longitude",
]


def screen_evidence_impact(row: pd.Series) -> dict:
    """Calculate an explainable impact screening score from observed evidence only.
    
    Does NOT assume unmeasured fields are zero. Evaluates multi-dimensional evidence completeness.
    """
    cas = row.get("reported_casualties")
    inj = row.get("injured_population")
    disp = row.get("displaced_population")
    aff = row.get("affected_population")
    loss = row.get("estimated_loss")
    lat = row.get("latitude")
    lon = row.get("longitude")
    desc = row.get("incident_description")

    harm_score = 0.0
    econ_score = 0.0
    evidence = []
    missing_fields = []

    # Evidence completeness across 7 key disaster reporting dimensions
    evidence_dimensions = [
        ("reported_casualties", cas),
        ("injured_population", inj),
        ("estimated_loss", loss),
        ("affected_population", aff),
        ("displaced_population", disp),
        ("geospatial_coordinates", lat if pd.notna(lat) and pd.notna(lon) else None),
        ("situational_narrative", desc if pd.notna(desc) and str(desc).strip() not in ("", "nan") else None),
    ]

    valid_dim_count = 0
    for name, val in evidence_dimensions:
        if pd.notna(val) and str(val).strip() not in ("", "nan", "null", "none"):
            valid_dim_count += 1
        else:
            missing_fields.append(name)

    completeness_pct = round(valid_dim_count / float(len(evidence_dimensions)) * 100.0, 1)

    # Missingness and numeric validity helpers
    def _is_missing(val):
        if pd.isna(val):
            return True
        s = str(val).strip().lower()
        return s in ("", "nan", "none", "null", "<na>")

    def _is_numeric(val):
        try:
            v = float(val)
            return not (math.isnan(v) or math.isinf(v))
        except (ValueError, TypeError):
            return False

    cas_missing = _is_missing(cas) or not _is_numeric(cas)
    inj_missing = _is_missing(inj) or not _is_numeric(inj)
    loss_missing = _is_missing(loss) or not _is_numeric(loss)

    cas_val = float(cas) if not cas_missing else None
    inj_val = float(inj) if not inj_missing else None
    loss_val = float(loss) if not loss_missing else None

    # 1. Human Harm Evidence
    has_fatalities = False
    has_injuries = False
    if cas_val is not None and cas_val > 0:
        has_fatalities = True
        harm_score += 40.0 + min((cas_val - 1.0) * 10.0, 15.0)
        evidence.append(f"{int(cas_val)} reported fatality" if cas_val == 1.0 else f"{int(cas_val)} reported fatalities")

    if inj_val is not None and inj_val > 0:
        has_injuries = True
        harm_score += min(inj_val * 15.0, 30.0)
        evidence.append(f"{int(inj_val)} injured persons")

    if not _is_missing(disp) and _is_numeric(disp) and float(disp) > 0:
        d_val = float(disp)
        harm_score += min(np.log10(d_val + 1.0) * 7.0, 20.0)
        evidence.append(f"{int(d_val):,} displaced population")

    if not _is_missing(aff) and _is_numeric(aff) and float(aff) > 0:
        a_val = float(aff)
        harm_score += min(np.log10(a_val + 1.0) * 4.0, 15.0)
        evidence.append(f"{int(a_val):,} affected population")

    # 2. Economic Damage Evidence
    if loss_val is not None and loss_val > 0:
        econ_score = min(np.log10(loss_val + 1.0) * 6.5, 50.0)
        evidence.append(f"${loss_val:,.0f} estimated property/crop loss")

    total_score = round(min(harm_score + econ_score, 100.0), 1)

    # 3. Categorize into explainable Relative Impact Tiers
    # Strictly distinguish missing/unassessed impact fields from verified zero
    if has_fatalities or harm_score >= 40.0 or econ_score >= 45.0 or total_score >= 70.0:
        tier = "Critical Impact"
        recommendation = "Immediate priority: dispatch emergency response teams and verify casualty relief."
    elif has_injuries or harm_score >= 15.0 or econ_score >= 35.0 or total_score >= 40.0:
        tier = "High Impact"
        recommendation = "High priority: coordinate medical aid, shelter assistance, or infrastructure repair."
    elif econ_score >= 20.0 or total_score >= 20.0:
        tier = "Moderate Impact"
        recommendation = "Moderate priority: support local recovery and monitor community recovery."
    elif total_score > 0.0:
        tier = "Minor Impact"
        recommendation = "Standard local assessment: log incident and verify minor repairs."
    else:
        # Total score is 0.0 (no positive harm or loss observed)
        # Check if ALL core impact dimensions are present and verified zero
        cas_verified_zero = (cas_val is not None and cas_val == 0.0)
        inj_verified_zero = (inj_val is not None and inj_val == 0.0)
        loss_verified_zero = (loss_val is not None and loss_val == 0.0)

        if cas_verified_zero and inj_verified_zero and loss_verified_zero:
            tier = "Negligible Observed Impact"
            evidence.append("Verified zero casualties, zero injuries, and $0 property/crop damage")
            recommendation = "Low priority: incident reported with verified zero casualties and zero losses."
        else:
            tier = "Unassessed / Incomplete Impact Evidence"
            missing_dims = []
            if cas_missing:
                missing_dims.append("casualties")
            if inj_missing:
                missing_dims.append("injuries")
            if loss_missing:
                missing_dims.append("economic damage")

            verified_dims = []
            if cas_verified_zero:
                verified_dims.append("casualties verified zero")
            if inj_verified_zero:
                verified_dims.append("injuries verified zero")
            if loss_verified_zero:
                verified_dims.append("losses verified zero ($0)")

            note = f"Impact unassessed/missing for: {', '.join(missing_dims)}"
            if verified_dims:
                note += f" ({', '.join(verified_dims)})"
            evidence.append(note)
            recommendation = f"Incident unassessed for {', '.join(missing_dims)}. Request local damage verification."

    return {
        "impact_screening_score": total_score,
        "impact_tier": tier,
        "contributing_evidence": evidence,
        "evidence_completeness_pct": completeness_pct,
        "missing_impact_fields": missing_fields,
        "human_review_recommendation": recommendation,
    }


def assess_csv_with_ml(
    df: pd.DataFrame,
    filename: str,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    client_id: Optional[str] = None,
    assessment_id: Optional[str] = None
) -> dict:
    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV contains no records."
        )

    # Standardize column headers
    df.columns = [str(col).strip() for col in df.columns]

    # Map canonical aliases (supporting both Member 1 combined fields and raw NOAA headers)
    aliases = {
        "incident_identifier": ["id", "incident_id", "incident_identifier", "EVENT_ID", "event_id"],
        "disaster_type": ["type", "disaster", "disaster_type", "EVENT_TYPE", "event_type"],
        "incident_location": ["zone", "location", "incident_location", "STATE", "state", "CZ_NAME", "cz_name"],
        "priority_level": ["priority", "priority_level", "PRIORITY_LEVEL"],
        "affected_population": ["affected", "affected_population", "AFFECTED_POPULATION"],
        "available_shelter_capacity": ["capacity", "available_shelter_capacity"],
        "current_shelter_occupancy": ["occupancy", "current_shelter_occupancy"],
        "incident_description": ["incident_description", "description", "EPISODE_NARRATIVE", "EVENT_NARRATIVE", "episode_narrative", "event_narrative"],
        "latitude": ["latitude", "lat", "BEGIN_LAT", "begin_lat", "LATITUDE"],
        "longitude": ["longitude", "lon", "BEGIN_LON", "begin_lon", "LONGITUDE"],
    }
    for canonical, alt_list in aliases.items():
        if canonical not in df.columns:
            for alt in alt_list:
                if alt in df.columns:
                    df[canonical] = df[alt]
                    break

    # Derive canonical combined fields from raw NOAA columns without double-counting
    # 1. Reported casualties (direct + indirect deaths)
    if "reported_casualties" not in df.columns:
        death_cols = [c for c in ["DEATHS_DIRECT", "DEATHS_INDIRECT", "deaths_direct", "deaths_indirect"] if c in df.columns]
        if death_cols:
            df["reported_casualties"] = df[death_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)

    # 2. Injured population (direct + indirect injuries)
    if "injured_population" not in df.columns:
        inj_cols = [c for c in ["INJURIES_DIRECT", "INJURIES_INDIRECT", "injuries_direct", "injuries_indirect"] if c in df.columns]
        if inj_cols:
            df["injured_population"] = df[inj_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)

    # 3. Estimated economic loss (property + crop damage)
    if "estimated_loss" not in df.columns:
        prop_col = next((c for c in ["DAMAGE_PROPERTY", "damage_property"] if c in df.columns), None)
        crop_col = next((c for c in ["DAMAGE_CROPS", "damage_crops"] if c in df.columns), None)
        if prop_col or crop_col:
            p_loss = df[prop_col].apply(parse_damage_string) if prop_col else pd.Series(np.nan, index=df.index)
            c_loss = df[crop_col].apply(parse_damage_string) if crop_col else pd.Series(np.nan, index=df.index)
            tot = p_loss.fillna(0) + c_loss.fillna(0)
            both_nan = p_loss.isna() & c_loss.isna()
            tot[both_nan] = np.nan
            df["estimated_loss"] = tot

    # Load Model Artifact: prefer newly retrained scikit-learn 1.9.1 hybrid artifact, fallback to base
    artifact = load_hybrid_artifact()
    artifact_type = "Hybrid (Feature-Aligned Isolation Forest + Evidence Impact Screener)"
    if not artifact or "model" not in artifact or "imputer" not in artifact or "scaler" not in artifact:
        artifact = load_isolation_forest_artifact()
        artifact_type = "Base Isolation Forest"
        if not artifact or "model" not in artifact or "imputer" not in artifact or "scaler" not in artifact:
            raise HTTPException(
                status_code=500,
                detail="Neither Hybrid nor Base Isolation Forest ML artifact could be loaded."
            )

    imputer = artifact["imputer"]
    scaler = artifact["scaler"]
    model = artifact["model"]
    feature_cols = artifact.get("feature_columns", MODEL_FEATURE_COLS)

    # Ensure all required model features exist in dataframe; if absent, populate as NaN
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Missing value tracking across the model's active features
    missing_per_feature = {col: int(df[col].isna().sum()) for col in feature_cols}
    total_missing_feature_cells = int(sum(missing_per_feature.values()))
    insufficient_mask = df[feature_cols].isna().all(axis=1)
    insufficient_data_count = int(insufficient_mask.sum())

    # Preprocess and execute Isolation Forest model
    X = df[feature_cols].copy()
    if "estimated_loss" in X.columns and (
        "log1p" in str(artifact.get("score_note", "")).lower()
        or "hybrid" in str(artifact.get("framework", "")).lower()
        or artifact_type.startswith("Hybrid")
    ):
        X["estimated_loss"] = np.log1p(X["estimated_loss"].clip(lower=0))

    try:
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)
        preds = model.predict(X_scaled)  # 1: Normal, -1: Anomaly
        scores = model.decision_function(X_scaled)  # Decision score; lower/negative = more unusual
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed while running Isolation Forest pipeline: {exc}"
        ) from exc

    df["anomaly_score"] = np.round(scores, 4)
    df["is_anomaly"] = (preds == -1)
    df["anomaly_status"] = np.where(preds == -1, "Unusual", "Normal")

    anomalies_detected = int((preds == -1).sum())
    anomaly_fraction = round(float((preds == -1).mean()), 4) if len(df) > 0 else 0.0
    anomaly_rate_pct = round(anomaly_fraction * 100.0, 2)

    # Apply Evidence-Based Impact Screening across all records
    screening_results = df.apply(screen_evidence_impact, axis=1)
    screening_df = pd.DataFrame(screening_results.tolist())
    df["impact_screening_score"] = screening_df["impact_screening_score"]
    df["impact_tier"] = screening_df["impact_tier"]
    df["contributing_evidence"] = screening_df["contributing_evidence"].apply(lambda ev: "; ".join(ev))
    df["evidence_completeness_pct"] = screening_df["evidence_completeness_pct"]
    df["human_review_recommendation"] = screening_df["human_review_recommendation"]

    impact_tier_counts = {str(k): int(v) for k, v in df["impact_tier"].value_counts().items()}
    critical_impact_events = int((df["impact_tier"] == "Critical Impact").sum())
    high_impact_events = int((df["impact_tier"] == "High Impact").sum())
    moderate_impact_events = int((df["impact_tier"] == "Moderate Impact").sum())
    negligible_impact_events = int((df["impact_tier"] == "Negligible Observed Impact").sum())
    unassessed_impact_events = int((df["impact_tier"] == "Unassessed / Incomplete Impact Evidence").sum())
    avg_completeness = round(float(df["evidence_completeness_pct"].mean()), 1) if len(df) > 0 else 0.0

    # Priority handling: preserve supplied priority, mark null/missing as Unknown
    if "priority_level" in df.columns:
        priority_map = {
            "critical": "Critical",
            "high": "High",
            "medium": "Medium",
            "low": "Low"
        }
        clean_priority = df["priority_level"].fillna("Unknown").astype(str).str.strip().apply(
            lambda v: "Unknown" if not v or str(v).lower() in ("nan", "none", "null", "<na>", "unknown")
            else priority_map.get(str(v).lower(), str(v))
        )
        df["priority_level"] = clean_priority
    else:
        df["priority_level"] = "Unknown"

    priority_counts = {str(k): int(v) for k, v in df["priority_level"].value_counts().items()}
    critical_count = int(priority_counts.get("Critical", 0))
    high_count = int(priority_counts.get("High", 0))
    medium_count = int(priority_counts.get("Medium", 0))
    low_count = int(priority_counts.get("Low", 0))
    unknown_count = int(priority_counts.get("Unknown", 0))

    # Shelter Capacity Analysis
    cap = df["available_shelter_capacity"].fillna(0) if "available_shelter_capacity" in df.columns else pd.Series(0, index=df.index)
    occ = df["current_shelter_occupancy"].fillna(0) if "current_shelter_occupancy" in df.columns else pd.Series(0, index=df.index)
    utilization_pct = np.where(cap > 0, np.round(100.0 * occ / cap, 1), 0.0)
    df["shelter_utilization_pct"] = utilization_pct

    shelter_overload_count = int(((cap > 0) & (utilization_pct >= 90.0)).sum())
    shelter_risk_count = int(((cap > 0) & (utilization_pct >= 75.0)).sum())

    # Evidence-based Overall Risk calculation strictly derived from verified impact evidence
    if critical_impact_events >= 2:
        overall_risk = "HIGH"
        overall_risk_reason = f"Verified evidence confirms {critical_impact_events} Critical Impact incidents with confirmed fatalities, severe injuries, or multi-million dollar damage."
    elif critical_impact_events >= 1 or high_impact_events >= 2:
        overall_risk = "MEDIUM"
        overall_risk_reason = f"Verified evidence indicates {critical_impact_events} Critical and {high_impact_events} High Impact incidents requiring prioritized field intervention."
    elif critical_impact_events == 0 and high_impact_events == 0:
        if unassessed_impact_events > (0.5 * len(df)):
            overall_risk = "Insufficient evidence"
            overall_risk_reason = f"{unassessed_impact_events} of {len(df)} records ({round(100.0 * unassessed_impact_events / max(1, len(df)), 1)}%) lack verified casualty or damage data. Overall risk cannot be reliably rated without ground-truth impact reports."
        elif negligible_impact_events + moderate_impact_events > 0:
            overall_risk = "LOW"
            overall_risk_reason = f"Verified evidence confirms zero fatalities and zero high-impact property loss across {negligible_impact_events + moderate_impact_events} reporting incidents."
        else:
            overall_risk = "Insufficient evidence"
            overall_risk_reason = "No confirmed casualty or economic impact metrics are available in the current dataset."
    else:
        overall_risk = "LOW"
        overall_risk_reason = "No critical or high-impact incidents observed in verified telemetry."

    # Dynamic Loss Coverage Calculation
    total_assessed = len(df)
    if total_assessed > 0 and "estimated_loss" in df.columns:
        loss_series = pd.to_numeric(df["estimated_loss"], errors="coerce")
        assessed_mask = loss_series.notna() & np.isfinite(loss_series)
        assessed_loss_count = int(assessed_mask.sum())
        unassessed_loss_count = total_assessed - assessed_loss_count
        coverage_pct = round((assessed_loss_count / total_assessed) * 100.0, 1) if total_assessed > 0 else 0.0
        unassessed_pct = round((unassessed_loss_count / total_assessed) * 100.0, 1) if total_assessed > 0 else 0.0
    else:
        assessed_loss_count = 0
        unassessed_loss_count = total_assessed
        coverage_pct = 0.0
        unassessed_pct = 100.0 if total_assessed > 0 else 0.0

    loss_coverage = {
        "total_records": total_assessed,
        "assessed_records": assessed_loss_count,
        "unassessed_records": unassessed_loss_count,
        "coverage_percentage": coverage_pct,
        "unassessed_percentage": unassessed_pct,
        "coverage_display": f"{coverage_pct}% Present ({unassessed_pct}% Unassessed)"
    }
    DATA_STORE["loss_coverage"] = loss_coverage

    # Actionable Data-Driven Recommendations (Hybrid: Evidence + Inventory + Anomaly + Data Verification)
    recs = []

    # 1. Critical & High Evidence-Based Impact Events (Rule-Based Impact Guidance + Inventory Linkage)
    high_impact_df = df[df["impact_tier"].isin(["Critical Impact", "High Impact"])].sort_values("impact_screening_score", ascending=False)
    for idx, row in high_impact_df.head(6).iterrows():
        loc = str(row.get("incident_location") or "Unspecified Location")
        dtype = str(row.get("disaster_type") or "Incident")
        inc_id = str(row.get("incident_identifier") or f"INC-{idx+1}")
        tier = str(row.get("impact_tier", "High Impact"))
        evidence_desc = str(row.get("contributing_evidence") or "Observed casualty or loss records")
        action_desc = str(row.get("human_review_recommendation") or f"Deploy emergency response team to {loc} and prioritize {dtype} incident.")

        # Incident-to-inventory matching for this incident
        match_info = database.match_inventory_for_location(loc)
        matched_shelters = match_info.get("matched_shelters", [])
        matched_resources = match_info.get("matched_resources", [])
        relevant_inv = matched_shelters + matched_resources

        recs.append({
            "incident_id": inc_id,
            "zone": loc,
            "location": loc,
            "disaster_type": dtype,
            "priority": "Critical" if tier == "Critical Impact" else "High",
            "impact_level": tier,
            "anomaly_status": "Baseline Observation",
            "guidance_type": "Rule-Based Impact Guidance",
            "supporting_evidence": evidence_desc,
            "action": f"[{tier}] {dtype} in {loc}: {evidence_desc}. {action_desc}",
            "recommended_action": action_desc,
            "relevant_inventory": relevant_inv,
            "match_details": match_info,
            "reason": f"Deterministic impact screener identified {tier} based on verified field reports ({evidence_desc}).",
            "source_type": "evidence rule",
            "source": "Evidence-Based Impact Screener",
            "uncertainty": "None (Verified casualty/loss observations)"
        })

    # 2. Flagged anomalies from Isolation Forest (Model Prediction / Statistical Anomaly)
    unusual_df = df[df["is_anomaly"]].sort_values("anomaly_score", ascending=True)
    for idx, row in unusual_df.head(5).iterrows():
        loc = str(row.get("incident_location") or "Unspecified Location")
        dtype = str(row.get("disaster_type") or "Incident")
        inc_id = str(row.get("incident_identifier") or f"INC-{idx+1}")
        ascore = safe_number(row.get("anomaly_score"), 0.0)
        recs.append({
            "incident_id": inc_id,
            "zone": loc,
            "location": loc,
            "disaster_type": dtype,
            "priority": "High",
            "impact_level": "Statistical Anomaly Review",
            "anomaly_status": "Statistical Outlier (Isolation Forest)",
            "guidance_type": "Model Prediction / Statistical Anomaly",
            "supporting_evidence": f"Isolation Forest multidimensional anomaly score: {ascore:.4f} (atypical feature profile)",
            "action": f"[Statistical Anomaly] Unusual pattern in {loc} ({dtype}) detected by Isolation Forest (score: {ascore}). Triage for uncharacteristic hazard conditions.",
            "recommended_action": f"Triage and conduct observational inspection in {loc} to verify if atypical hazard conditions or reporting artifacts exist.",
            "reason": "Multivariate statistical anomaly detected by Isolation Forest. Indicates atypical pattern requiring verification, not verified disaster severity.",
            "source_type": "anomaly review",
            "source": "Isolation Forest Anomaly Detection",
            "uncertainty": "Atypical pattern detected; impact severity unconfirmed without physical inspection"
        })

    # 3. Data Quality Verification (For Unassessed / Incomplete Impact Evidence in severe hazard categories)
    unassessed_df = df[df["impact_tier"] == "Unassessed / Incomplete Impact Evidence"]
    severe_hazard_mask = unassessed_df["disaster_type"].str.lower().str.contains("tornado|flood|wildfire|hail|hurricane|blizzard", na=False)
    severe_unassessed = unassessed_df[severe_hazard_mask]
    for idx, row in severe_unassessed.head(2).iterrows():
        loc = str(row.get("incident_location") or "Unspecified Location")
        dtype = str(row.get("disaster_type") or "Incident")
        inc_id = str(row.get("incident_identifier") or f"INC-{idx+1}")
        evidence_desc = str(row.get("contributing_evidence") or "Impact evidence unassessed")
        recs.append({
            "incident_id": inc_id,
            "zone": loc,
            "location": loc,
            "disaster_type": dtype,
            "priority": "Medium",
            "impact_level": "Unassessed / Incomplete Impact Evidence",
            "anomaly_status": "Telemetry Gap",
            "guidance_type": "Data Quality Verification",
            "supporting_evidence": evidence_desc,
            "action": f"[Data Quality Verification] {dtype} in {loc}: {evidence_desc}. Request local field verification.",
            "recommended_action": f"Request local field verification in {loc} to submit confirmed casualty, injury, and property loss tallies.",
            "reason": f"Incident occurred in a severe hazard category ({dtype}) but lacks verified impact reports ({evidence_desc}).",
            "source_type": "data-quality verification",
            "source": "Telemetry Data Verification",
            "uncertainty": "Impact evidence unrecorded in telemetry"
        })

    # 4. Shelters near or exceeding capacity (ONLY if real shelter capacity is present in data)
    has_real_shelters = "available_shelter_capacity" in df.columns and (pd.to_numeric(df["available_shelter_capacity"], errors="coerce").fillna(0) > 0).any()
    if has_real_shelters:
        overloaded_df = df[(df["available_shelter_capacity"] > 0) & (df["shelter_utilization_pct"] >= 85.0)].sort_values("shelter_utilization_pct", ascending=False)
        for idx, row in overloaded_df.head(4).iterrows():
            loc = str(row.get("incident_location") or "Unspecified Location")
            u = row["shelter_utilization_pct"]
            c = int(row.get("available_shelter_capacity", 0))
            o = int(row.get("current_shelter_occupancy", 0))
            recs.append({
                "incident_id": f"SH-{idx+1:02d}",
                "zone": loc,
                "location": loc,
                "disaster_type": "Shelter Capacity",
                "priority": "Critical" if u >= 100.0 else "High",
                "impact_level": "Facility Capacity Stress",
                "anomaly_status": "Operational Threshold",
                "guidance_type": "Rule-Based Capacity Guidance",
                "supporting_evidence": f"Facility occupancy is {u}% ({o}/{c} occupants)",
                "action": f"Coordinate shelter overflow in {loc}; facility utilization is at {u}%.",
                "recommended_action": f"Activate secondary relief shelter or coordinate overflow transport for {loc}.",
                "reason": f"Facility occupancy ({u}%) exceeds safe operational reserve threshold.",
                "source_type": "shelter capacity rule",
                "source": "Shelter Capacity Monitor",
                "uncertainty": "None (Reported capacity and occupancy metrics verified)"
            })

    if not recs:
        recs.append({
            "incident_id": "SYS-01",
            "zone": "All Monitored Locations",
            "location": "All Monitored Locations",
            "disaster_type": "All Hazards",
            "priority": "Low",
            "impact_level": "Negligible Observed Impact",
            "anomaly_status": "Normal Baseline",
            "guidance_type": "Operational Baseline Guidance",
            "supporting_evidence": "All reported observations are within standard baseline parameters.",
            "action": "All reported observations are within standard baseline parameters. Continue normal observational monitoring.",
            "recommended_action": "Maintain routine situational awareness and standard monitoring schedules.",
            "reason": "No critical impacts, statistical anomalies, or capacity risks detected in verified data.",
            "source_type": "evidence rule",
            "source": "System Assessment",
            "uncertainty": "None (Observations within baseline)"
        })

    # Deterministic priority ordering: Critical -> High -> Medium -> Low
    p_order = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4}
    recs.sort(key=lambda item: (p_order.get(item["priority"], 99), str(item.get("incident_id", ""))))

    # Assign recommendation IDs
    for r_idx, r_item in enumerate(recs, 1):
        if "recommendation_id" not in r_item:
            r_item["recommendation_id"] = f"REC-{r_idx:02d}"

    # Update in-memory DATA_STORE for Member 2 frontend dashboard compatibility
    updated_incidents = []
    for idx, row in df.iterrows():
        score = float(row.get("impact_screening_score", 30.0))
        p_label = str(row.get("priority_level", "Unknown"))
        if p_label == "Unknown" and row["impact_tier"] not in ("Negligible Observed Impact", "Unassessed / Incomplete Impact Evidence", "Negligible Impact", "Indeterminate / Insufficient Evidence"):
            p_display = "Critical" if row["impact_tier"] == "Critical Impact" else ("High" if row["impact_tier"] == "High Impact" else ("Medium" if row["impact_tier"] == "Moderate Impact" else "Low"))
        else:
            p_display = p_label

        updated_incidents.append({
            "id": str(row.get("incident_identifier") or f"INC-{idx+1}"),
            "zone": str(row.get("incident_location") or f"Zone {idx+1}"),
            "type": str(row.get("disaster_type") or "Incident"),
            "severity": 5 if row["impact_tier"] == "Critical Impact" else (4 if row["impact_tier"] == "High Impact" else 3),
            "affected": safe_number(row.get("affected_population"), 0),
            "vulnerability": safe_number(row.get("vulnerability"), 0),
            "distance": safe_number(row.get("distance"), 0),
            "priority_score": score,
            "priority": p_display,
            "official_priority": p_label,
            "anomaly_score": safe_number(row.get("anomaly_score"), 0.0),
            "is_anomaly": bool(row["is_anomaly"]),
            "impact_screening_score": safe_number(row.get("impact_screening_score"), 0.0),
            "impact_tier": str(row.get("impact_tier", "Unknown")),
            "estimated_loss": None if pd.isna(row.get("estimated_loss")) else float(row.get("estimated_loss")),
            "reported_casualties": int(safe_number(row.get("reported_casualties"), 0)),
            "injured_population": int(safe_number(row.get("injured_population"), 0)),
            "contributing_evidence": str(row.get("contributing_evidence") or "")
        })
    DATA_STORE["incidents"] = updated_incidents

    # Update shelter records in DATA_STORE (empty if dataset lacks real shelter data)
    shelter_records = []
    valid_shelters = df[df["available_shelter_capacity"] > 0] if "available_shelter_capacity" in df.columns else pd.DataFrame()
    for idx, row in valid_shelters.head(20).iterrows():
        c = safe_number(row.get("available_shelter_capacity"), 0)
        o = safe_number(row.get("current_shelter_occupancy"), 0)
        u = round(o / c * 100.0, 1) if c > 0 else 0.0
        st = "Critical" if u >= 90.0 else ("Near Capacity" if u >= 75.0 else "Available")
        shelter_records.append({
            "id": f"SH-{idx+1:02d}",
            "name": f"Shelter {row.get('incident_location', idx+1)}",
            "zone": str(row.get("incident_location") or f"Zone {idx+1}"),
            "capacity": int(c),
            "occupancy": int(o),
            "utilization": u,
            "available": max(int(c - o), 0),
            "status": st
        })
    DATA_STORE["shelters"] = shelter_records

    # Update resource records in DATA_STORE (empty if dataset lacks operational resource tracking)
    resource_records = []
    if "required" in df.columns and "available" in df.columns:
        for idx, row in df[df["required"].notna() & df["available"].notna()].head(20).iterrows():
            resource_records.append({
                "zone": str(row.get("incident_location") or row.get("zone") or f"Zone {idx+1}"),
                "type": str(row.get("resource_type") or row.get("type") or "Supplies"),
                "required": safe_number(row.get("required"), 0),
                "available": safe_number(row.get("available"), 0),
                "unit": str(row.get("unit") or "units")
            })
    DATA_STORE["resources"] = resource_records

    # Reconciled anomaly list: store all flagged anomalies in DATA_STORE (no truncation to head 20)
    anomaly_list = []
    for idx, row in unusual_df.iterrows():
        ascore = safe_number(row.get("anomaly_score"), 0.0)
        loc = str(row.get("incident_location") or "Zone")
        dtype = str(row.get("disaster_type") or "Incident")
        inc_id = str(row.get("incident_identifier") or f"REC-{idx+1}")
        has_real_cap = bool("available_shelter_capacity" in row and safe_number(row.get("available_shelter_capacity"), 0) > 0)
        anomaly_list.append({
            "id": inc_id,
            "type": dtype,
            "shelter": f"Record {inc_id} ({loc})",
            "zone": loc,
            "utilization": safe_number(row.get("shelter_utilization_pct"), 0.0),
            "anomaly": True,
            "anomaly_score": ascore,
            "status": "Unusual",
            "has_real_utilization": has_real_cap,
            "contributing_evidence": str(row.get("contributing_evidence") or "")
        })
    DATA_STORE["anomalies"] = anomaly_list

    # Update recommendations in DATA_STORE with verified assessment recommendations
    DATA_STORE["recommendations"] = recs

    # Set correlation and assessment metadata in DATA_STORE
    clean_stem = Path(filename).stem
    job_id = clean_stem.replace("_processed", "")
    req_id = request_id or str(uuid.uuid4())
    corr_id = correlation_id or str(uuid.uuid4())
    asm_id = assessment_id or f"ASM-{uuid.uuid4().hex[:8].upper()}"
    cli_id = client_id or "Unspecified Client (No X-Source-System header)"

    DATA_STORE["current_job_id"] = job_id
    DATA_STORE["current_request_id"] = req_id
    DATA_STORE["current_correlation_id"] = corr_id
    DATA_STORE["current_assessment_id"] = asm_id
    DATA_STORE["current_client_id"] = cli_id
    DATA_STORE["current_filename"] = filename
    DATA_STORE["total_assessed_records"] = len(df)
    DATA_STORE["anomalies_detected"] = anomalies_detected
    DATA_STORE["critical_impact_events"] = critical_impact_events
    DATA_STORE["dispatch_priority_critical_count"] = critical_count
    DATA_STORE["overall_risk"] = overall_risk
    DATA_STORE["overall_risk_reason"] = overall_risk_reason
    DATA_STORE["loss_coverage"] = loss_coverage
    DATA_STORE["data_quality_report"] = {
        "economic_loss_coverage": loss_coverage
    }

    update_timestamp()

    # Save processed CSV with standard naming for GET /api/processed-results
    RECEIVED_DIR = Path(__file__).resolve().parent / "received_datasets"
    RECEIVED_DIR.mkdir(parents=True, exist_ok=True)
    processed_filename = f"{clean_stem}_processed.csv" if not clean_stem.endswith("_processed") else f"{clean_stem}.csv"
    processed_path = RECEIVED_DIR / processed_filename
    df.to_csv(processed_path, index=False)

    # Sample evaluated records for response
    sample_columns = [
        col for col in [
            "incident_identifier", "disaster_type", "incident_location",
            "priority_level", "affected_population", "reported_casualties",
            "injured_population", "estimated_loss",
            "available_shelter_capacity", "current_shelter_occupancy",
            "shelter_utilization_pct", "anomaly_score", "is_anomaly", "anomaly_status",
            "impact_screening_score", "impact_tier", "contributing_evidence",
            "human_review_recommendation"
        ] if col in df.columns
    ]
    sample_records = json.loads(df[sample_columns].head(10).to_json(orient="records"))

    return {
        "status": "success",
        "job_id": job_id,
        "request_id": req_id,
        "correlation_id": corr_id,
        "assessment_id": asm_id,
        "client_id": cli_id,
        "model_version": "2.1.0-hybrid",
        "assessment_timestamp": datetime.now().isoformat(),
        "processing_status": "COMPLETED",
        "filename": filename,
        "processed_file_saved": processed_filename,
        "total_records": len(df),
        "valid_records_count": len(df) - insufficient_data_count,
        "insufficient_data_records_count": insufficient_data_count,
        "summary": {
            "total_incidents": len(df),
            "total_assessed_records": len(df),
            "critical_incidents": critical_count,
            "critical_impact_events": critical_impact_events,
            "dispatch_priority_critical_count": critical_count,
            "high_priority_incidents": high_count,
            "medium_priority_incidents": medium_count,
            "low_priority_incidents": low_count,
            "unknown_priority_incidents": unknown_count,
            "anomalies_detected": anomalies_detected,
            "anomalies": anomalies_detected,
            "anomaly_rate_pct": anomaly_rate_pct,
            "shelter_risks": shelter_risk_count,
            "shelter_overload_count": shelter_overload_count,
            "overall_risk": overall_risk,
            "overall_risk_reason": overall_risk_reason,
            "loss_coverage": loss_coverage,
            "data_quality_report": {
                "economic_loss_coverage": loss_coverage
            },
            "shelter_summary": {
                "monitored_shelter_records": int((cap > 0).sum()),
                "at_risk_shelters": shelter_risk_count,
                "overloaded_shelters": shelter_overload_count,
                **database.get_shelter_summary_metrics()
            },
            "resource_summary": database.get_resource_summary_metrics(),
            "last_updated": DATA_STORE["last_updated"],
            "job_id": job_id,
            "request_id": req_id,
            "correlation_id": corr_id,
            "assessment_id": asm_id,
            "client_id": cli_id
        },
        "ml_assessment": {
            "model_name": artifact_type,
            "model_status": "Operational (Evaluated via scikit-learn artifact)",
            "trained_feature_columns": feature_cols,
            "missing_feature_cells": total_missing_feature_cells,
            "missing_cells_per_feature": missing_per_feature,
            "anomalies_flagged": anomalies_detected,
            "anomaly_fraction": anomaly_fraction,
            "anomaly_rate_pct": anomaly_rate_pct,
            "score_interpretation": "Isolation Forest decision_function; lower/negative values indicate unusual multi-dimensional conditions.",
            "priority_model_status": "Ground truth preserved; missing values marked 'Unknown'. Priority classifier candidate (~24.5% accuracy vs 25.7% baseline) not deployed for operational classification.",
            "data_quality_warnings": [
                "Isolation Forest anomaly flags represent statistical outliers for human review, not verified disaster ground truth.",
                "Candidate priority classifier was omitted from operational scoring to prevent misleading predictions."
            ]
        },
        "hybrid_assessment": {
            "framework": "Hybrid Multi-Attribute Disaster Assessment",
            "status": "Operational",
            "anomaly_detection": {
                "model": "Isolation Forest",
                "active_features": feature_cols,
                "anomalies_flagged": anomalies_detected,
                "anomaly_rate_pct": anomaly_rate_pct,
                "interpretation": "Unsupervised multidimensional review signals."
            },
            "evidence_based_impact_screening": {
                "status": "Operational",
                "impact_tier_distribution": impact_tier_counts,
                "average_evidence_completeness_pct": avg_completeness,
                "critical_impact_events": critical_impact_events,
                "high_impact_events": high_impact_events,
                "interpretation": "Deterministic impact indicators computed strictly from observed casualties, injuries, and estimated losses."
            },
            "supervised_priority_classification": {
                "status": "Candidate Only (Evaluated on 57,481 Verified NOAA Records; Not Deployed)",
                "ground_truth_status": "NOAA Storm Events contains 100% Unknown/missing municipal priority labels. Official emergency priority ground truth does not exist.",
                "leakage_safe_hazard_benchmark": {
                    "target": "Significant Hazard Impact (Project-defined operational rule: Fatalities > 0 OR Injuries > 0 OR Loss >= $10k)",
                    "dataset_filter": "Excluded 14,879 records with zero casualties and missing damage (NaN) to prevent false-negative corruption. Total validated records: 57,481.",
                    "split_strategy": "StratifiedGroupKFold on EPISODE_ID (Zero storm overlap between train and test).",
                    "input_features": ["EVENT_TYPE", "STATE", "MONTH_NAME", "CZ_TYPE", "BEGIN_LAT", "BEGIN_LON", "MAGNITUDE"],
                    "leakage_safeguard": "Strictly zero casualties, injuries, losses, or damage proxies in input features. Preprocessing fit only on training partitions.",
                    "held_out_test_metrics_11495_records": {
                        "dummy_majority_class": {"accuracy": 0.9075, "balanced_accuracy": 0.5000, "macro_f1": 0.4758},
                        "logistic_regression_balanced": {"accuracy": 0.7701, "balanced_accuracy": 0.7377, "macro_f1": 0.6097, "recall_class_1": 0.6980, "group_cv_bal_acc": 0.7440},
                        "random_forest_balanced": {"accuracy": 0.7662, "balanced_accuracy": 0.7403, "macro_f1": 0.6081, "recall_class_1": 0.7084, "group_cv_bal_acc": 0.7530},
                        "hist_gradient_boosting_default_t050": {"accuracy": 0.8157, "balanced_accuracy": 0.7700, "macro_f1": 0.6539, "precision_class_1": 0.2949, "recall_class_1": 0.7140, "false_positives": 1815},
                        "hist_gradient_boosting_tuned_t060": {"accuracy": 0.8659, "balanced_accuracy": 0.7487, "macro_f1": 0.6892, "precision_class_1": 0.3645, "recall_class_1": 0.6049, "false_positives": 1121, "ap_pr_auc": 0.4779, "precision_gain_pct": 23.6, "false_positive_reduction_pct": 38.2}
                    },
                    "scientific_finding": "Threshold tuning (t=0.60) on validation data improved precision from 29.49% to 36.45% and reduced false alarms by 38.2% on the held-out test set while satisfying recall >= 60%. Supervised model remains a Candidate Only; dispatch operations rely deterministically on verified Evidence-Based Impact Screening."
                }
            }
        },
        "shelter_summary": {
            "monitored_shelter_records": int((cap > 0).sum()),
            "at_risk_shelters": shelter_risk_count,
            "overloaded_shelters": shelter_overload_count,
            **database.get_shelter_summary_metrics()
        },
        "resource_summary": database.get_resource_summary_metrics(),
        "loss_coverage": loss_coverage,
        "data_quality_report": {
            "economic_loss_coverage": loss_coverage
        },
        "sample_evaluated_records": sample_records,
        "recommendations": recs,
        "shelters": DATA_STORE["shelters"],
        "resources": DATA_STORE["resources"],
        "assessment": generate_assessment()
    }


@app.post("/api/receive-file")
async def receive_file(
    file: UploadFile = File(...),
    x_source_system: Optional[str] = Header(None),
    x_client_id: Optional[str] = Header(None),
    x_correlation_id: Optional[str] = Header(None),
):
    request_ts = datetime.now().isoformat()
    req_id = str(uuid.uuid4())
    corr_id = str(x_correlation_id).strip() if x_correlation_id else str(uuid.uuid4())
    
    # Documented client identification: derived from explicit integration header without guessing from IP
    if x_source_system and str(x_source_system).strip():
        cli_id = str(x_source_system).strip()
    elif x_client_id and str(x_client_id).strip():
        cli_id = str(x_client_id).strip()
    else:
        cli_id = "Unspecified Client (No X-Source-System header)"

    raw_filename = file.filename or "disaster_dataset.csv"
    clean_stem = Path(raw_filename).stem
    job_id = clean_stem.replace("_processed", "")
    asm_id = f"ASM-{uuid.uuid4().hex[:8].upper()}"

    filename = (file.filename or "").lower()

    try:
        content = await file.read()

        if not content:
            record_api_activity(
                timestamp=request_ts,
                client_id=cli_id,
                filename=raw_filename,
                request_id=req_id,
                job_id=job_id,
                assessment_id=asm_id,
                correlation_id=corr_id,
                records_count=0,
                processing_status="FAILED",
                http_status=400,
                error_message="The uploaded file is empty."
            )
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        RECEIVED_DIR = Path(__file__).resolve().parent / "received_datasets"
        RECEIVED_DIR.mkdir(parents=True, exist_ok=True)

        original_filename = Path(file.filename or "disaster_dataset.csv").name
        saved_path = RECEIVED_DIR / original_filename
        saved_path.write_bytes(content)

        print(f"Received file saved at: {saved_path}")

        # Process CSV files with genuine ML assessment pipeline
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content))
            except Exception as exc:
                record_api_activity(
                    timestamp=request_ts,
                    client_id=cli_id,
                    filename=original_filename,
                    request_id=req_id,
                    job_id=job_id,
                    assessment_id=asm_id,
                    correlation_id=corr_id,
                    records_count=0,
                    processing_status="FAILED",
                    http_status=400,
                    error_message=f"Could not parse uploaded CSV: {exc}"
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not parse uploaded CSV: {exc}"
                ) from exc

            result = assess_csv_with_ml(
                df,
                original_filename,
                request_id=req_id,
                correlation_id=corr_id,
                client_id=cli_id,
                assessment_id=asm_id
            )

            record_api_activity(
                timestamp=request_ts,
                client_id=cli_id,
                filename=original_filename,
                request_id=req_id,
                job_id=job_id,
                assessment_id=asm_id,
                correlation_id=corr_id,
                records_count=len(df),
                processing_status="COMPLETED",
                http_status=200,
                error_message=None
            )
            return result

        elif filename.endswith(".json"):
            payload = json.loads(content.decode("utf-8"))

            if isinstance(payload, list):
                payload = {"records": payload}

            if not isinstance(payload, dict):
                record_api_activity(
                    timestamp=request_ts,
                    client_id=cli_id,
                    filename=original_filename,
                    request_id=req_id,
                    job_id=job_id,
                    assessment_id=asm_id,
                    correlation_id=corr_id,
                    records_count=0,
                    processing_status="FAILED",
                    http_status=400,
                    error_message="JSON must contain an object or array."
                )
                raise HTTPException(
                    status_code=400,
                    detail="JSON must contain an object or array."
                )
            result = receive(payload)
            record_api_activity(
                timestamp=request_ts,
                client_id=cli_id,
                filename=original_filename,
                request_id=req_id,
                job_id=job_id,
                assessment_id=asm_id,
                correlation_id=corr_id,
                records_count=len(payload.get("records", [])),
                processing_status="COMPLETED",
                http_status=200,
                error_message=None
            )
            return result

        else:
            record_api_activity(
                timestamp=request_ts,
                client_id=cli_id,
                filename=raw_filename,
                request_id=req_id,
                job_id=job_id,
                assessment_id=asm_id,
                correlation_id=corr_id,
                records_count=0,
                processing_status="FAILED",
                http_status=415,
                error_message="Only CSV and JSON files are supported."
            )
            raise HTTPException(
                status_code=415,
                detail="Only CSV and JSON files are supported."
            )

    except HTTPException:
        raise

    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        record_api_activity(
            timestamp=request_ts,
            client_id=cli_id,
            filename=raw_filename,
            request_id=req_id,
            job_id=job_id,
            assessment_id=asm_id,
            correlation_id=corr_id,
            records_count=0,
            processing_status="FAILED",
            http_status=400,
            error_message=f"Could not parse uploaded file: {exc}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse uploaded file: {exc}"
        ) from exc

    finally:
        await file.close()


# ============================================================
# API ACTIVITY LOGS
# ============================================================

@app.get("/api/activity-logs")
def get_activity_logs(limit: int = 20, correlation_id: Optional[str] = None):
    logs = API_ACTIVITY_LOGS
    if correlation_id:
        logs = [entry for entry in logs if entry.get("correlation_id") == correlation_id]
    return {
        "status": "success",
        "count": len(logs[:limit]),
        "logs": logs[:limit]
    }


# ============================================================
# RESET DATA
# ============================================================

@app.post("/api/reset")
def reset_data():

    DATA_STORE["incidents"] = deepcopy(
        INITIAL_INCIDENTS
    )

    database.reset_shelters_to_demo()
    database.reset_resources_to_demo()
    DATA_STORE["shelters"] = database.get_all_shelters()
    DATA_STORE["resources"] = database.get_all_resources()

    DATA_STORE["anomalies"] = None
    DATA_STORE["recommendations"] = None
    DATA_STORE["current_job_id"] = None
    DATA_STORE["current_request_id"] = None
    DATA_STORE["current_correlation_id"] = None
    DATA_STORE["current_assessment_id"] = None
    DATA_STORE["current_client_id"] = None
    DATA_STORE["current_filename"] = None
    DATA_STORE["total_assessed_records"] = None
    DATA_STORE["anomalies_detected"] = None
    DATA_STORE["critical_impact_events"] = None
    DATA_STORE["dispatch_priority_critical_count"] = None
    DATA_STORE["overall_risk"] = None
    DATA_STORE["overall_risk_reason"] = None
    DATA_STORE["loss_coverage"] = None

    update_timestamp()

    return {
        "status": "reset",
        "message": "Disaster data restored to initial state",
        "last_updated": DATA_STORE["last_updated"]
    }


# ============================================================
# SHELTER MANAGEMENT ENDPOINTS
# ============================================================

@app.get("/api/shelters")
def get_shelters():
    return {
        "status": "success",
        "shelters": database.get_all_shelters(),
        "summary": database.get_shelter_summary_metrics(),
        "last_updated": DATA_STORE["last_updated"]
    }


@app.post("/api/shelters")
def create_shelter(payload: Dict[str, Any]):
    try:
        shelter = database.create_shelter(payload)
        update_timestamp()
        DATA_STORE["shelters"] = database.get_all_shelters()
        return {
            "status": "success",
            "message": f"Shelter '{shelter['shelter_name']}' created successfully.",
            "shelter": shelter
        }
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/api/shelters/{shelter_id}")
def update_shelter(shelter_id: str, payload: Dict[str, Any]):
    try:
        shelter = database.update_shelter(shelter_id, payload)
        update_timestamp()
        DATA_STORE["shelters"] = database.get_all_shelters()
        return {
            "status": "success",
            "message": f"Shelter '{shelter_id}' updated successfully.",
            "shelter": shelter
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/shelters/{shelter_id}")
def delete_shelter(shelter_id: str):
    deleted = database.delete_shelter(shelter_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Shelter '{shelter_id}' not found.")
    update_timestamp()
    DATA_STORE["shelters"] = database.get_all_shelters()
    return {
        "status": "success",
        "message": f"Shelter '{shelter_id}' deleted successfully.",
        "deleted_id": shelter_id
    }


@app.post("/api/shelters/reset")
def reset_shelters():
    shelters = database.reset_shelters_to_demo()
    update_timestamp()
    DATA_STORE["shelters"] = shelters
    return {
        "status": "success",
        "message": "Shelters successfully reset to verified demo dataset.",
        "shelters": shelters,
        "summary": database.get_shelter_summary_metrics()
    }


@app.get("/api/shelters/template")
def download_shelters_template():
    csv_content = database.generate_shelter_csv_template()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=shelters_template.csv"}
    )


@app.post("/api/shelters/import")
async def import_shelters(file: UploadFile = File(...)):
    try:
        content = await file.read()
        csv_text = content.decode("utf-8-sig", errors="replace")
        imported = database.import_shelters_csv(csv_text)
        update_timestamp()
        DATA_STORE["shelters"] = database.get_all_shelters()
        return {
            "status": "success",
            "message": f"Successfully imported {len(imported)} shelter records.",
            "imported_count": len(imported),
            "shelters": database.get_all_shelters(),
            "summary": database.get_shelter_summary_metrics()
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Import error: {str(exc)}")


# ============================================================
# RESOURCE MANAGEMENT ENDPOINTS
# ============================================================

@app.get("/api/resources")
def get_resources():
    return {
        "status": "success",
        "resources": database.get_all_resources(),
        "summary": database.get_resource_summary_metrics(),
        "last_updated": DATA_STORE["last_updated"]
    }


@app.post("/api/resources")
def create_resource(payload: Dict[str, Any]):
    try:
        res = database.create_resource(payload)
        update_timestamp()
        DATA_STORE["resources"] = database.get_all_resources()
        return {
            "status": "success",
            "message": f"Resource '{res['resource_type']}' created successfully.",
            "resource": res
        }
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/api/resources/{resource_id}")
def update_resource(resource_id: str, payload: Dict[str, Any]):
    try:
        res = database.update_resource(resource_id, payload)
        update_timestamp()
        DATA_STORE["resources"] = database.get_all_resources()
        return {
            "status": "success",
            "message": f"Resource '{resource_id}' updated successfully.",
            "resource": res
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/resources/{resource_id}")
def delete_resource(resource_id: str):
    deleted = database.delete_resource(resource_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Resource '{resource_id}' not found.")
    update_timestamp()
    DATA_STORE["resources"] = database.get_all_resources()
    return {
        "status": "success",
        "message": f"Resource '{resource_id}' deleted successfully.",
        "deleted_id": resource_id
    }


@app.post("/api/resources/reset")
def reset_resources():
    resources = database.reset_resources_to_demo()
    update_timestamp()
    DATA_STORE["resources"] = resources
    return {
        "status": "success",
        "message": "Emergency resources successfully reset to verified demo dataset.",
        "resources": resources,
        "summary": database.get_resource_summary_metrics()
    }


@app.get("/api/resources/template")
def download_resources_template():
    csv_content = database.generate_resource_csv_template()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=resources_template.csv"}
    )


@app.post("/api/resources/import")
async def import_resources(file: UploadFile = File(...)):
    try:
        content = await file.read()
        csv_text = content.decode("utf-8-sig", errors="replace")
        imported = database.import_resources_csv(csv_text)
        update_timestamp()
        DATA_STORE["resources"] = database.get_all_resources()
        return {
            "status": "success",
            "message": f"Successfully imported {len(imported)} resource records.",
            "imported_count": len(imported),
            "resources": database.get_all_resources(),
            "summary": database.get_resource_summary_metrics()
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Import error: {str(exc)}")


# ============================================================
# INCIDENT-TO-INVENTORY MATCHING ENDPOINT
# ============================================================

@app.get("/api/inventory/match")
def match_inventory(
    location: Optional[str] = Query(None),
    incident_id: Optional[str] = Query(None)
):
    """Match incident evidence with relevant shelter and resource inventories."""
    inc_obj = None
    target_loc = location

    if incident_id:
        for inc in DATA_STORE.get("incidents", []):
            if str(inc.get("id")) == str(incident_id):
                inc_obj = inc
                if not target_loc:
                    target_loc = inc.get("zone") or inc.get("incident_location")
                break

    if not target_loc:
        raise HTTPException(
            status_code=400,
            detail="Must provide either a 'location' parameter or a valid 'incident_id'."
        )

    matched = database.match_inventory_for_location(target_loc)
    
    # Calculate operational totals for matching items
    avail_beds = sum(s.get("available_beds", 0) for s in matched.get("matched_shelters", []))
    shortages_count = sum(1 for r in matched.get("matched_resources", []) if r.get("shortage", 0) > 0)
    
    warnings = []
    for s in matched.get("matched_shelters", []):
        if s.get("status") == "Full":
            warnings.append(f"WARNING: Matched shelter '{s['shelter_name']}' is at 100% capacity ({s['occupancy']}/{s['capacity']} beds). DO NOT route evacuees here.")
        elif s.get("status") == "Near Capacity":
            warnings.append(f"ADVISORY: Matched shelter '{s['shelter_name']}' is near capacity ({s['utilization_percentage']}%). Monitor intake.")

    return {
        "status": "success",
        "incident_id": incident_id,
        "location": target_loc,
        "incident_details": inc_obj,
        "matched_shelters": matched.get("matched_shelters", []),
        "matched_resources": matched.get("matched_resources", []),
        "available_beds_total": avail_beds,
        "resource_shortages_count": shortages_count,
        "warnings": warnings,
        "match_type": matched.get("match_type"),
        "match_confidence": matched.get("match_confidence"),
        "message": matched.get("message"),
        "operational_disclaimer": "Decision support matching based on approximate normalized text. Not an automated 911 dispatch."
    }

# ============================================================
# ML RESULTS API
# ============================================================

ML_RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "received_datasets"
    / "member2_ml_artifacts"
)


def get_supervised_benchmark_path() -> Optional[Path]:
    p1 = Path(__file__).resolve().parent / "received_datasets" / "member2_hybrid_artifacts" / "supervised_benchmark_report.json"
    if p1.exists():
        return p1
    p2 = Path(__file__).resolve().parent.parent / "member2_hybrid_artifacts" / "supervised_benchmark_report.json"
    if p2.exists():
        return p2
    return None


def get_legacy_benchmark_path() -> Optional[Path]:
    p1 = Path(__file__).resolve().parent / "received_datasets" / "member2_ml_artifacts" / "evaluation_report.json"
    if p1.exists():
        return p1
    p2 = Path(__file__).resolve().parent.parent / "member2_ml_artifacts" / "evaluation_report.json"
    if p2.exists():
        return p2
    return None


@app.get("/api/ml-results")
def get_ml_results():
    """Return comprehensive ML results and evaluation reports.

    Explicitly separates:
    1. Active real NOAA assessment results (tied to active dataset, assessment ID,
       real 500-record anomaly counts, and unavailable shelter inventory).
    2. Historical offline benchmark candidates (HistGradientBoosting on 57k NOAA records,
       and legacy mock Random Forest, both clearly labeled as offline candidates/not deployed).
    """
    total_records = DATA_STORE.get("total_assessed_records")
    if total_records is None:
        total_records = len(DATA_STORE.get("incidents", []))

    anomalies_detected = DATA_STORE.get("anomalies_detected")
    if anomalies_detected is None:
        anomalies_detected = sum(bool(item.get("anomaly", False)) for item in (DATA_STORE.get("anomalies") or []))

    anomaly_fraction = round(float(anomalies_detected) / max(1, total_records), 4) if total_records else 0.0
    anomaly_rate_pct = round(anomaly_fraction * 100.0, 2)

    active_filename = DATA_STORE.get("current_filename") or "Current Assessment"
    active_job_id = DATA_STORE.get("current_job_id")
    active_assessment_id = DATA_STORE.get("current_assessment_id")
    critical_impact_events = DATA_STORE.get("critical_impact_events", 0)
    dispatch_priority_critical_count = DATA_STORE.get("dispatch_priority_critical_count", 0)
    overall_risk = DATA_STORE.get("overall_risk") or "HIGH"
    overall_risk_reason = DATA_STORE.get("overall_risk_reason") or ""

    # Inspect active CSV on disk if available to get column, missing-cell, and priority counts
    received_dir = Path(__file__).resolve().parent / "received_datasets"
    active_csv_path = received_dir / active_filename if active_filename else None
    column_count = 24
    missing_cells = 0
    priority_counts = {}
    if active_csv_path and active_csv_path.exists():
        try:
            sample_df = pd.read_csv(active_csv_path)
            column_count = len(sample_df.columns)
            missing_cells = int(sample_df.isna().sum().sum())
            if "priority_level" in sample_df.columns:
                priority_counts = sample_df["priority_level"].fillna("Unknown").astype(str).str.strip().value_counts().to_dict()
        except Exception:
            pass

    if not priority_counts:
        # Fall back to checking incidents or default to Unknown for NOAA
        for inc in DATA_STORE.get("incidents", []):
            raw_p = str(inc.get("priority_level", "Unknown")).strip()
            priority_counts[raw_p] = priority_counts.get(raw_p, 0) + 1
    if not priority_counts and total_records:
        priority_counts = {"Unknown": total_records}

    # Load offline HistGradientBoosting benchmark report
    sb_path = get_supervised_benchmark_path()
    supervised_benchmark_data = None
    if sb_path and sb_path.exists():
        try:
            with open(sb_path, "r", encoding="utf-8") as f:
                sb_report = json.load(f)
            val_eval = sb_report.get("held_out_test_evaluation", {}).get("tuned_threshold_t_0_60", {})
            supervised_benchmark_data = {
                "model_name": "HistGradientBoostingClassifier",
                "status": "Candidate Only - Not Deployed",
                "deployment_status": "Candidate Only (Not Deployed)",
                "evaluation_title": sb_report.get("evaluation_title", "Supervised HistGradientBoosting Precision Optimization Benchmark"),
                "evaluated_dataset": "NOAA Storm Events 2025 (57,481 verified records, 14,879 unassessed records excluded)",
                "split_strategy": "StratifiedGroupKFold on EPISODE_ID (zero storm overlap between train and test)",
                "input_features": ["EVENT_TYPE", "STATE", "MONTH_NAME", "CZ_TYPE", "BEGIN_LAT", "BEGIN_LON", "MAGNITUDE"],
                "leakage_safeguard": "Strictly pre-event hazard attributes; zero casualty or loss proxies in features",
                "tuned_threshold": sb_report.get("threshold_tuning", {}).get("selected_threshold", 0.60),
                "held_out_test_metrics": {
                    "accuracy": val_eval.get("accuracy", 0.8659),
                    "balanced_accuracy": val_eval.get("balanced_accuracy", 0.7487),
                    "precision_class_1": val_eval.get("precision", 0.3645),
                    "recall_class_1": val_eval.get("recall", 0.6049),
                    "f1_class_1": val_eval.get("f1", 0.4552),
                    "macro_f1": val_eval.get("macro_f1", 0.6892),
                    "average_precision_pr_auc": val_eval.get("average_precision", 0.4779),
                    "false_positives": val_eval.get("confusion_matrix", [[0, 1121]])[0][1] if "confusion_matrix" in val_eval else 1121,
                },
                "recommendation": "Candidate Only (Not Deployed). Supervised model exhibits a 63.5% false alarm rate on pre-event attributes; operational dispatch must remain driven deterministically by verified Evidence-Based Impact Screening."
            }
        except Exception as exc:
            supervised_benchmark_data = {"status": "Error", "detail": f"Could not parse supervised benchmark: {exc}"}

    # Load offline legacy Random Forest benchmark report
    leg_path = get_legacy_benchmark_path()
    legacy_benchmark_data = None
    if leg_path and leg_path.exists():
        try:
            with open(leg_path, "r", encoding="utf-8") as f:
                leg_report = json.load(f)
            legacy_benchmark_data = {
                "model_name": "Random Forest Classifier",
                "status": "Historical Benchmark - Obsolete / Not Validated",
                "evaluated_dataset": f"Historical Synthetic Benchmark ({Path(leg_report.get('dataset', '')).name}, {leg_report.get('rows', 10000)} rows)",
                "accuracy": leg_report.get("random_forest_accuracy", 0.245),
                "baseline_accuracy": leg_report.get("majority_baseline_accuracy", 0.257),
                "balanced_accuracy": leg_report.get("balanced_accuracy", 0.2441),
                "classification_report": leg_report.get("classification_report"),
                "confusion_matrix": leg_report.get("confusion_matrix"),
                "confusion_matrix_labels": leg_report.get("confusion_matrix_labels"),
                "historical_anomaly_detection": {
                    "model": "Isolation Forest",
                    "anomaly_count": leg_report.get("anomaly_count", 500),
                    "anomaly_fraction": leg_report.get("anomaly_fraction", 0.05),
                    "note": "Historical benchmark computed on synthetic 10k CSV with mock shelter columns. In real NOAA datasets, physical shelter inventory is unavailable."
                },
                "recommendation": "Obsolete mock benchmark. Do not deploy; evaluated on synthetic priority distribution before real NOAA data integration."
            }
        except Exception as exc:
            legacy_benchmark_data = {"status": "Error", "detail": f"Could not parse legacy benchmark: {exc}"}

    active_assessment = {
        "assessment_id": active_assessment_id,
        "job_id": active_job_id,
        "filename": active_filename,
        "total_records": total_records,
        "provenance": "Current Active NOAA Assessment",
        "overall_risk": overall_risk,
        "overall_risk_reason": overall_risk_reason,
        "critical_impact_events": critical_impact_events,
        "dispatch_priority_critical_count": dispatch_priority_critical_count,
        "priority_class_counts": priority_counts,
        "priority_ground_truth_status": "NOAA Storm Events contains 100% Unknown/missing municipal priority labels; official CAD emergency priority ground truth does not exist.",
        "anomaly_detection": {
            "model": "Isolation Forest",
            "framework": "Feature-Aligned Isolation Forest (72k Historical NOAA)",
            "status": "Operational",
            "anomaly_count": anomalies_detected,
            "anomaly_fraction": anomaly_fraction,
            "anomaly_rate_pct": anomaly_rate_pct,
            "active_features": [
                "reported_casualties",
                "injured_population",
                "estimated_loss",
                "latitude",
                "longitude"
            ],
            "interpretation": "Unsupervised multidimensional anomaly detection; lower/negative decision scores indicate unusual feature profiles for human verification."
        },
        "impact_screening": {
            "model": "Evidence-Based Impact Screener",
            "status": "Operational",
            "critical_impact_events": critical_impact_events,
            "interpretation": "Deterministic multi-attribute impact screener evaluated strictly on observed casualties, injuries, and property/crop losses."
        },
        "shelter_inventory": {
            "status": "Unavailable",
            "monitored_shelter_records": 0,
            "shelter_utilization_summary_pct": None,
            "note": "Verified physical shelter inventory is not available in meteorological disaster telemetry."
        }
    }

    dataset_summary = {
        "filename": active_filename,
        "rows": total_records,
        "columns": column_count,
        "priority_class_counts": priority_counts,
        "missing_cells": missing_cells,
        "provenance": "Current Active NOAA Assessment",
        "assessment_id": active_assessment_id,
        "job_id": active_job_id
    }

    model_summary = {
        "name": "Supervised Priority Classification Candidate",
        "status": "Candidate Only - Not Deployed",
        "deployed": False,
        "operational_role": "None - Operational dispatch relies deterministically on Evidence-Based Impact Screening",
        "ground_truth_status": "NOAA Storm Events contains 100% Unknown/missing municipal priority labels; official CAD emergency priority ground truth does not exist.",
        "accuracy": supervised_benchmark_data["held_out_test_metrics"]["accuracy"] if supervised_benchmark_data and "held_out_test_metrics" in supervised_benchmark_data else None,
        "balanced_accuracy": supervised_benchmark_data["held_out_test_metrics"]["balanced_accuracy"] if supervised_benchmark_data and "held_out_test_metrics" in supervised_benchmark_data else None,
        "recommendation": "Supervised classification is an offline research benchmark and is NOT deployed. Emergency operations rely deterministically on verified Evidence-Based Impact Screening."
    }

    anomaly_detection_summary = {
        "model": "Isolation Forest",
        "status": "Operational",
        "anomaly_count": anomalies_detected,
        "anomaly_fraction": anomaly_fraction,
        "anomaly_rate_pct": anomaly_rate_pct,
        "active_features": [
            "reported_casualties",
            "injured_population",
            "estimated_loss",
            "latitude",
            "longitude"
        ],
        "shelter_utilization_summary_pct": None,
        "shelter_status": "Unavailable",
        "shelter_note": "No verified shelter inventory data exists in NOAA telemetry; shelter utilization stats are unavailable."
    }

    data_quality_warnings = [
        "NOAA records are historical meteorological reports; missing casualty or damage values represent unrecorded telemetry, not confirmed zero impact.",
        "Shelter inventory feeds are not present in NOAA Storm Events telemetry and are marked unavailable.",
        "Isolation Forest anomaly flags represent statistical outliers for human review, not verified emergency ground truth.",
        "Supervised classification models remain unvalidated offline candidates; dispatch decisions must not rely on unvalidated predictions."
    ]

    return {
        "status": "success",
        "active_assessment": active_assessment,
        "dataset": dataset_summary,
        "model": model_summary,
        "anomaly_detection": anomaly_detection_summary,
        "historical_offline_benchmarks": {
            "supervised_hist_gradient_boosting": supervised_benchmark_data,
            "historical_legacy_random_forest": legacy_benchmark_data
        },
        "data_quality_warnings": data_quality_warnings
    }
    
#==============================================
# PROCESSED DATA API
# ============================================================

@app.get("/api/processed-results")
def get_processed_results(job_id: Optional[str] = None, filename: Optional[str] = None):
    received_dir = Path(__file__).resolve().parent / "received_datasets"

    target_file = None
    if job_id:
        clean_id = job_id.replace("_processed", "")
        matches = sorted(
            received_dir.glob(f"*{clean_id}*.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if matches:
            target_file = matches[0]
    elif filename:
        matches = sorted(
            received_dir.glob(filename),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if matches:
            target_file = matches[0]
    elif DATA_STORE.get("current_job_id"):
        clean_id = str(DATA_STORE["current_job_id"]).replace("_processed", "")
        matches = sorted(
            received_dir.glob(f"*{clean_id}*.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if matches:
            target_file = matches[0]

    if not target_file:
        csv_files = sorted(
            received_dir.glob("*_processed.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if not csv_files:
            raise HTTPException(
                status_code=404,
                detail="No processed CSV file found. Integrate a dataset first.",
            )
        target_file = csv_files[0]

    try:
        df = pd.read_csv(target_file).fillna("")
        return {
            "status": "success",
            "filename": target_file.name,
            "job_id": target_file.stem.replace("_processed", ""),
            "total_records": len(df),
            "columns": df.columns.tolist(),
            "records": json.loads(df.to_json(orient="records")),
        }
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read processed dataset: {exc}",
        ) from exc


# ============================================================
# STARTUP RESTORATION
# ============================================================

def restore_latest_assessment_from_disk():
    """On server restart, restore the latest valid processed assessment from disk.
    Never silently substitute hardcoded mock incidents when real assessment data exists.
    """
    received_dir = Path(__file__).resolve().parent / "received_datasets"
    if not received_dir.exists():
        return
    candidate_files = sorted(
        [
            f for f in received_dir.glob("job-*_processed.csv")
            if not f.name.startswith("test_")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if candidate_files:
        latest_file = candidate_files[0]
        try:
            df = pd.read_csv(latest_file)
            if not df.empty:
                assess_csv_with_ml(df, latest_file.name)
        except Exception as exc:
            print(f"Notice: Auto-restore from {latest_file.name} skipped: {exc}")

# Auto-restore latest assessment on startup
restore_latest_assessment_from_disk()

