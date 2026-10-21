"""SQLite Database & Inventory Persistence Module for Member 2.

Manages persistent shelter and emergency-resource inventories across application restarts.
Calculates dynamic operational metrics (available beds, utilization %, supply shortages/surpluses)
and enforces input validation.
"""

import sqlite3
import math
import csv
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

DB_PATH = Path(__file__).resolve().parent / "inventory.db"

DEMO_DATA_SOURCE = "DEMO DATA — NOT LIVE MUNICIPAL INVENTORY"
USER_DATA_SOURCE = "User-Supplied Inventory"

# Seed data adhering to prompt: multiple locations, varying occupancy levels
# including near-capacity (>=80% and <100%) and full (100%) shelters
DEMO_SHELTERS = [
    {
        "shelter_id": "SH-01",
        "shelter_name": "San Juan Coastal Evacuation Facility",
        "location": "San Juan",
        "capacity": 1000,
        "occupancy": 650,
        "data_source": DEMO_DATA_SOURCE,
        "notes": "Primary coastal relief shelter with emergency power generation."
    },
    {
        "shelter_id": "SH-02",
        "shelter_name": "Okanogan Civic Center Relief Depot",
        "location": "Okanogan",
        "capacity": 500,
        "occupancy": 440,
        "data_source": DEMO_DATA_SOURCE,
        "notes": "Near capacity due to regional wildfire evacuations."
    },
    {
        "shelter_id": "SH-03",
        "shelter_name": "Northern Panhandle Mountain Shelter",
        "location": "Northern Panhandle",
        "capacity": 300,
        "occupancy": 300,
        "data_source": DEMO_DATA_SOURCE,
        "notes": "Facility at 100% capacity; cannot accept additional evacuees."
    },
    {
        "shelter_id": "SH-04",
        "shelter_name": "Highland Regional High School Gym",
        "location": "Highlands",
        "capacity": 600,
        "occupancy": 210,
        "data_source": DEMO_DATA_SOURCE,
        "notes": "Active intake point with large athletic field staging."
    },
    {
        "shelter_id": "SH-05",
        "shelter_name": "Metro Valley Community Shelter",
        "location": "Metro Valley",
        "capacity": 400,
        "occupancy": 375,
        "data_source": DEMO_DATA_SOURCE,
        "notes": "Near capacity (93.8% utilization); monitor for overflow."
    }
]

# Seed resources: at least one shortage, one sufficient, one surplus
DEMO_RESOURCES = [
    {
        "resource_id": "RES-01",
        "resource_type": "Drinking Water",
        "location": "Okanogan",
        "required_quantity": 15000.0,
        "available_quantity": 6000.0,
        "unit": "Liters",
        "data_source": DEMO_DATA_SOURCE,
        "notes": "High deficit for wildfire evacuation zone; supply convoy en route."
    },
    {
        "resource_id": "RES-02",
        "resource_type": "Medical Trauma Kits",
        "location": "San Juan",
        "required_quantity": 500.0,
        "available_quantity": 180.0,
        "unit": "kits",
        "data_source": DEMO_DATA_SOURCE,
        "notes": "Coastal surge medical kits needed for waterborne rescue stations."
    },
    {
        "resource_id": "RES-03",
        "resource_type": "Emergency Food Rations",
        "location": "Northern Panhandle",
        "required_quantity": 3000.0,
        "available_quantity": 3000.0,
        "unit": "packs",
        "data_source": DEMO_DATA_SOURCE,
        "notes": "Stock on hand perfectly matches current shelter headcount."
    },
    {
        "resource_id": "RES-04",
        "resource_type": "Thermal Wool Blankets",
        "location": "Highlands",
        "required_quantity": 1000.0,
        "available_quantity": 1600.0,
        "unit": "units",
        "data_source": DEMO_DATA_SOURCE,
        "notes": "Surplus reserve stock available to transfer to neighboring zones."
    },
    {
        "resource_id": "RES-05",
        "resource_type": "Emergency Power Generators",
        "location": "Metro Valley",
        "required_quantity": 40.0,
        "available_quantity": 15.0,
        "unit": "units",
        "data_source": DEMO_DATA_SOURCE,
        "notes": "Severe shortage (25 units missing) for mobile triage units."
    }
]


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables and seed demo records if tables are empty."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shelters (
                shelter_id TEXT PRIMARY KEY,
                shelter_name TEXT NOT NULL,
                location TEXT NOT NULL,
                capacity INTEGER NOT NULL CHECK (capacity > 0),
                occupancy INTEGER NOT NULL CHECK (occupancy >= 0),
                last_updated TEXT NOT NULL,
                data_source TEXT NOT NULL,
                notes TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resources (
                resource_id TEXT PRIMARY KEY,
                resource_type TEXT NOT NULL,
                location TEXT NOT NULL,
                required_quantity REAL NOT NULL CHECK (required_quantity >= 0),
                available_quantity REAL NOT NULL CHECK (available_quantity >= 0),
                unit TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                data_source TEXT NOT NULL,
                notes TEXT
            )
        """)
        conn.commit()

        # Seed shelters if empty
        cursor.execute("SELECT COUNT(*) FROM shelters")
        if cursor.fetchone()[0] == 0:
            ts = datetime.now().isoformat()
            for s in DEMO_SHELTERS:
                cursor.execute("""
                    INSERT INTO shelters (shelter_id, shelter_name, location, capacity, occupancy, last_updated, data_source, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (s["shelter_id"], s["shelter_name"], s["location"], s["capacity"], s["occupancy"], ts, s["data_source"], s.get("notes", "")))
            conn.commit()

        # Seed resources if empty
        cursor.execute("SELECT COUNT(*) FROM resources")
        if cursor.fetchone()[0] == 0:
            ts = datetime.now().isoformat()
            for r in DEMO_RESOURCES:
                cursor.execute("""
                    INSERT INTO resources (resource_id, resource_type, location, required_quantity, available_quantity, unit, last_updated, data_source, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (r["resource_id"], r["resource_type"], r["location"], r["required_quantity"], r["available_quantity"], r["unit"], ts, r["data_source"], r.get("notes", "")))
            conn.commit()


# ============================================================
# SHELTER OPERATIONS & VALIDATIONS
# ============================================================

def calculate_shelter_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate dynamic shelter operational metrics from ground-truth capacity & occupancy."""
    try:
        cap = int(row["capacity"])
        occ = int(row["occupancy"])
    except (TypeError, ValueError):
        return {
            **row,
            "available_beds": 0,
            "utilization_percentage": 0.0,
            "utilization": 0.0,
            "status": "Invalid Data"
        }

    if cap <= 0 or occ < 0 or occ > cap:
        status = "Invalid Data"
        avail = 0
        u_pct = 0.0
    else:
        avail = cap - occ
        u_pct = round((occ / cap) * 100.0, 1)
        if u_pct >= 100.0:
            status = "Full"
        elif u_pct >= 80.0:
            status = "Near Capacity"
        else:
            status = "Available"

    return {
        "shelter_id": str(row["shelter_id"]),
        "id": str(row["shelter_id"]),
        "shelter_name": str(row["shelter_name"]),
        "name": str(row["shelter_name"]),
        "location": str(row["location"]),
        "zone": str(row["location"]),
        "capacity": cap,
        "occupancy": occ,
        "available_beds": avail,
        "available": avail,
        "utilization_percentage": u_pct,
        "utilization": u_pct,
        "status": status,
        "data_source": str(row.get("data_source", DEMO_DATA_SOURCE)),
        "is_demo": DEMO_DATA_SOURCE in str(row.get("data_source", "")),
        "last_updated": str(row.get("last_updated", "")),
        "notes": str(row.get("notes") or "")
    }


def get_all_shelters() -> List[Dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM shelters ORDER BY shelter_id")
        rows = cursor.fetchall()
        return [calculate_shelter_fields(dict(r)) for r in rows]


def get_shelter_by_id(shelter_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM shelters WHERE shelter_id = ?", (shelter_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return calculate_shelter_fields(dict(row))


def create_shelter(data: Dict[str, Any]) -> Dict[str, Any]:
    init_db()
    shelter_id = str(data.get("shelter_id") or "").strip()
    shelter_name = str(data.get("shelter_name") or data.get("name") or "").strip()
    location = str(data.get("location") or data.get("zone") or "").strip()
    
    if not shelter_name:
        raise ValueError("Shelter Name is required.")
    if not location:
        raise ValueError("Location / Zone is required.")

    # Auto-generate unique shelter_id if left blank
    if not shelter_id:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT shelter_id FROM shelters")
            existing_ids = {str(row["shelter_id"]) for row in cursor.fetchall()}
            max_num = 0
            for sid in existing_ids:
                if sid.startswith("SH-") and sid[3:].isdigit():
                    try:
                        max_num = max(max_num, int(sid[3:]))
                    except ValueError:
                        pass
            candidate = f"SH-{max_num + 1:02d}"
            while candidate in existing_ids:
                max_num += 1
                candidate = f"SH-{max_num + 1:02d}"
            shelter_id = candidate

    try:
        capacity = int(data.get("capacity"))
    except (TypeError, ValueError):
        raise ValueError("Capacity must be a positive integer.")
    if capacity <= 0:
        raise ValueError("Capacity must be a positive integer greater than zero.")

    try:
        occupancy = int(data.get("occupancy", 0))
    except (TypeError, ValueError):
        raise ValueError("Occupancy must be a non-negative integer.")
    if occupancy < 0:
        raise ValueError("Occupancy cannot be negative.")
    if occupancy > capacity:
        raise ValueError(f"Occupancy ({occupancy}) cannot exceed certified capacity ({capacity}).")

    data_source = str(data.get("data_source") or USER_DATA_SOURCE).strip()
    notes = str(data.get("notes") or "").strip()
    ts = datetime.now().isoformat()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT shelter_id FROM shelters WHERE shelter_id = ?", (shelter_id,))
        if cursor.fetchone():
            raise ValueError(f"Shelter ID '{shelter_id}' already exists. Use a unique ID.")
        
        cursor.execute("""
            INSERT INTO shelters (shelter_id, shelter_name, location, capacity, occupancy, last_updated, data_source, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (shelter_id, shelter_name, location, capacity, occupancy, ts, data_source, notes))
        conn.commit()

    return get_shelter_by_id(shelter_id)


def update_shelter(shelter_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    init_db()
    existing = get_shelter_by_id(shelter_id)
    if not existing:
        raise KeyError(f"Shelter '{shelter_id}' not found.")

    shelter_name = str(data.get("shelter_name") or data.get("name") or existing["shelter_name"]).strip()
    location = str(data.get("location") or data.get("zone") or existing["location"]).strip()
    
    cap_val = data.get("capacity", existing["capacity"])
    occ_val = data.get("occupancy", existing["occupancy"])

    try:
        capacity = int(cap_val)
    except (TypeError, ValueError):
        raise ValueError("Capacity must be a positive integer.")
    if capacity <= 0:
        raise ValueError("Capacity must be greater than zero.")

    try:
        occupancy = int(occ_val)
    except (TypeError, ValueError):
        raise ValueError("Occupancy must be a non-negative integer.")
    if occupancy < 0:
        raise ValueError("Occupancy cannot be negative.")
    if occupancy > capacity:
        raise ValueError(f"Occupancy ({occupancy}) cannot exceed capacity ({capacity}).")

    data_source = str(data.get("data_source") or existing["data_source"]).strip()
    notes = str(data.get("notes") if "notes" in data else existing.get("notes", "")).strip()
    ts = datetime.now().isoformat()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE shelters
            SET shelter_name = ?, location = ?, capacity = ?, occupancy = ?, last_updated = ?, data_source = ?, notes = ?
            WHERE shelter_id = ?
        """, (shelter_name, location, capacity, occupancy, ts, data_source, notes, shelter_id))
        conn.commit()

    return get_shelter_by_id(shelter_id)


def delete_shelter(shelter_id: str) -> bool:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM shelters WHERE shelter_id = ?", (shelter_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted


def reset_shelters_to_demo() -> List[Dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM shelters")
        ts = datetime.now().isoformat()
        for s in DEMO_SHELTERS:
            cursor.execute("""
                INSERT INTO shelters (shelter_id, shelter_name, location, capacity, occupancy, last_updated, data_source, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (s["shelter_id"], s["shelter_name"], s["location"], s["capacity"], s["occupancy"], ts, s["data_source"], s.get("notes", "")))
        conn.commit()
    return get_all_shelters()


# ============================================================
# RESOURCE OPERATIONS & VALIDATIONS
# ============================================================

def calculate_resource_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate dynamic supply shortages, surpluses, and statuses."""
    try:
        req = float(row["required_quantity"])
        avail = float(row["available_quantity"])
    except (TypeError, ValueError):
        return {
            **row,
            "shortage": 0.0,
            "surplus": 0.0,
            "availability_percent": 0.0,
            "status": "Invalid Data"
        }

    if req < 0 or avail < 0:
        status = "Invalid Data"
        shortage = 0.0
        surplus = 0.0
        avail_pct = 0.0
    else:
        shortage = round(max(req - avail, 0.0), 2)
        surplus = round(max(avail - req, 0.0), 2)
        if req > 0:
            avail_pct = round((avail / req) * 100.0, 1)
        else:
            avail_pct = 100.0 if avail > 0 else 0.0

        if avail < req:
            status = "Shortage"
        elif avail > req:
            status = "Surplus"
        else:
            status = "Sufficient"

    return {
        "resource_id": str(row["resource_id"]),
        "id": str(row["resource_id"]),
        "resource_type": str(row["resource_type"]),
        "type": str(row["resource_type"]),
        "location": str(row["location"]),
        "zone": str(row["location"]),
        "required_quantity": req,
        "required": req,
        "available_quantity": avail,
        "available": avail,
        "unit": str(row["unit"]),
        "shortage": shortage,
        "surplus": surplus,
        "availability_percent": avail_pct,
        "status": status,
        "data_source": str(row.get("data_source", DEMO_DATA_SOURCE)),
        "is_demo": DEMO_DATA_SOURCE in str(row.get("data_source", "")),
        "last_updated": str(row.get("last_updated", "")),
        "notes": str(row.get("notes") or "")
    }


def get_all_resources() -> List[Dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM resources ORDER BY resource_id")
        rows = cursor.fetchall()
        return [calculate_resource_fields(dict(r)) for r in rows]


def get_resource_by_id(resource_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM resources WHERE resource_id = ?", (resource_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return calculate_resource_fields(dict(row))


def create_resource(data: Dict[str, Any]) -> Dict[str, Any]:
    init_db()
    resource_id = str(data.get("resource_id") or "").strip()
    resource_type = str(data.get("resource_type") or data.get("type") or "").strip()
    location = str(data.get("location") or data.get("zone") or "").strip()
    unit = str(data.get("unit") or "").strip()

    if not resource_type:
        raise ValueError("Resource Type is required.")
    if not location:
        raise ValueError("Location / Zone is required.")
    if not unit:
        raise ValueError("Unit of measure is required.")

    # Auto-generate unique resource_id if left blank
    if not resource_id:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT resource_id FROM resources")
            existing_ids = {str(row["resource_id"]) for row in cursor.fetchall()}
            max_num = 0
            for rid in existing_ids:
                if rid.startswith("RES-") and rid[4:].isdigit():
                    try:
                        max_num = max(max_num, int(rid[4:]))
                    except ValueError:
                        pass
            candidate = f"RES-{max_num + 1:02d}"
            while candidate in existing_ids:
                max_num += 1
                candidate = f"RES-{max_num + 1:02d}"
            resource_id = candidate

    try:
        req = float(data.get("required_quantity", data.get("required", 0)))
    except (TypeError, ValueError):
        raise ValueError("Required quantity must be a non-negative number.")
    if req < 0:
        raise ValueError("Required quantity cannot be negative.")

    try:
        avail = float(data.get("available_quantity", data.get("available", 0)))
    except (TypeError, ValueError):
        raise ValueError("Available quantity must be a non-negative number.")
    if avail < 0:
        raise ValueError("Available quantity cannot be negative.")

    data_source = str(data.get("data_source") or USER_DATA_SOURCE).strip()
    notes = str(data.get("notes") or "").strip()
    ts = datetime.now().isoformat()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT resource_id FROM resources WHERE resource_id = ?", (resource_id,))
        if cursor.fetchone():
            raise ValueError(f"Resource ID '{resource_id}' already exists. Use a unique ID.")

        cursor.execute("""
            INSERT INTO resources (resource_id, resource_type, location, required_quantity, available_quantity, unit, last_updated, data_source, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (resource_id, resource_type, location, req, avail, unit, ts, data_source, notes))
        conn.commit()

    return get_resource_by_id(resource_id)


def update_resource(resource_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    init_db()
    existing = get_resource_by_id(resource_id)
    if not existing:
        raise KeyError(f"Resource '{resource_id}' not found.")

    resource_type = str(data.get("resource_type") or data.get("type") or existing["resource_type"]).strip()
    location = str(data.get("location") or data.get("zone") or existing["location"]).strip()
    unit = str(data.get("unit") or existing["unit"]).strip()

    req_val = data.get("required_quantity", data.get("required", existing["required_quantity"]))
    avail_val = data.get("available_quantity", data.get("available", existing["available_quantity"]))

    try:
        req = float(req_val)
    except (TypeError, ValueError):
        raise ValueError("Required quantity must be a non-negative number.")
    if req < 0:
        raise ValueError("Required quantity cannot be negative.")

    try:
        avail = float(avail_val)
    except (TypeError, ValueError):
        raise ValueError("Available quantity must be a non-negative number.")
    if avail < 0:
        raise ValueError("Available quantity cannot be negative.")

    data_source = str(data.get("data_source") or existing["data_source"]).strip()
    notes = str(data.get("notes") if "notes" in data else existing.get("notes", "")).strip()
    ts = datetime.now().isoformat()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE resources
            SET resource_type = ?, location = ?, required_quantity = ?, available_quantity = ?, unit = ?, last_updated = ?, data_source = ?, notes = ?
            WHERE resource_id = ?
        """, (resource_type, location, req, avail, unit, ts, data_source, notes, resource_id))
        conn.commit()

    return get_resource_by_id(resource_id)


def delete_resource(resource_id: str) -> bool:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM resources WHERE resource_id = ?", (resource_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted


def reset_resources_to_demo() -> List[Dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM resources")
        ts = datetime.now().isoformat()
        for r in DEMO_RESOURCES:
            cursor.execute("""
                INSERT INTO resources (resource_id, resource_type, location, required_quantity, available_quantity, unit, last_updated, data_source, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (r["resource_id"], r["resource_type"], r["location"], r["required_quantity"], r["available_quantity"], r["unit"], ts, r["data_source"], r.get("notes", "")))
        conn.commit()
    return get_all_resources()


# ============================================================
# INCIDENT-TO-INVENTORY INTEGRATION & MATCHING
# ============================================================

def match_inventory_for_location(location_str: str) -> Dict[str, Any]:
    """Find candidate shelters and resources in stored inventory matching location keywords.
    Clearly marks matches as approximate text matches.
    """
    if not location_str or not str(location_str).strip():
        return {
            "matched_location": "Unspecified Location",
            "match_type": "None",
            "matched_shelters": [],
            "matched_resources": [],
            "message": "No location provided for inventory matching. Request field verification."
        }

    target_tokens = [tok.lower().strip(" ,.-/") for tok in str(location_str).split() if len(tok.strip(" ,.-/")) >= 3]
    all_shelters = get_all_shelters()
    all_resources = get_all_resources()

    matched_shelters = []
    for s in all_shelters:
        s_loc = s["location"].lower()
        if any(tok in s_loc or s_loc in tok for tok in target_tokens):
            matched_shelters.append(s)

    matched_resources = []
    for r in all_resources:
        r_loc = r["location"].lower()
        if any(tok in r_loc or r_loc in tok for tok in target_tokens):
            matched_resources.append(r)

    has_matches = bool(matched_shelters or matched_resources)
    return {
        "query_location": location_str,
        "match_type": "Approximate Normalized Text Match" if has_matches else "No Match",
        "match_confidence": "Approximate Location Heuristic (Not Verified GPS Coordinates)" if has_matches else "None",
        "matched_shelters": matched_shelters,
        "matched_resources": matched_resources,
        "message": (
            f"Found {len(matched_shelters)} shelter(s) and {len(matched_resources)} resource(s) matching '{location_str}'."
            if has_matches else
            f"No suitable shelter or resource inventory found for location '{location_str}'. Request local logistics verification."
        )
    }


def get_shelter_summary_metrics() -> Dict[str, Any]:
    """Calculate aggregate metrics across all stored shelters."""
    shelters = get_all_shelters()
    total_shelters = len(shelters)
    total_capacity = sum(s["capacity"] for s in shelters)
    total_occupancy = sum(s["occupancy"] for s in shelters)
    total_available_beds = sum(s["available_beds"] for s in shelters)
    overall_utilization = round((total_occupancy / total_capacity * 100.0), 1) if total_capacity > 0 else 0.0
    near_capacity_count = sum(1 for s in shelters if s["status"] == "Near Capacity")
    full_count = sum(1 for s in shelters if s["status"] == "Full")
    available_count = sum(1 for s in shelters if s["status"] == "Available")
    return {
        "total_shelters": total_shelters,
        "total_capacity": total_capacity,
        "total_occupancy": total_occupancy,
        "total_available_beds": total_available_beds,
        "overall_utilization_pct": overall_utilization,
        "near_capacity_count": near_capacity_count,
        "full_count": full_count,
        "available_count": available_count
    }


def get_resource_summary_metrics() -> Dict[str, Any]:
    """Calculate aggregate metrics across all stored emergency resources."""
    resources = get_all_resources()
    total_resources = len(resources)
    resource_types_tracked = len(set(r["resource_type"] for r in resources))
    shortage_count = sum(1 for r in resources if r["status"] in ("Shortage", "Critical Shortage") or r.get("shortage", 0) > 0)
    surplus_count = sum(1 for r in resources if r["status"] == "Surplus" or r.get("surplus", 0) > 0)
    sufficient_count = sum(1 for r in resources if r["status"] == "Sufficient" and r.get("shortage", 0) == 0 and r.get("surplus", 0) == 0)
    shortage_items = sorted([r for r in resources if r.get("shortage", 0) > 0], key=lambda x: x.get("shortage", 0), reverse=True)
    return {
        "total_resources": total_resources,
        "resource_types_tracked": resource_types_tracked,
        "shortage_count": shortage_count,
        "surplus_count": surplus_count,
        "sufficient_count": sufficient_count,
        "largest_shortages": shortage_items[:5]
    }


# ============================================================
# CSV TEMPLATE AND IMPORT/EXPORT UTILITIES
# ============================================================

def generate_shelter_csv_template() -> str:
    """Return a standard downloadable CSV template for shelter bulk imports."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["shelter_id", "shelter_name", "location", "capacity", "occupancy", "notes"])
    writer.writerow(["SH-IMPORT-01", "Regional Sports Complex Shelter", "Metro Valley", 800, 320, "Backup generator available"])
    writer.writerow(["SH-IMPORT-02", "Central Community College Hall", "San Juan", 450, 150, "Full ADA accessibility"])
    return output.getvalue()


def import_shelters_csv(csv_text: str) -> List[Dict[str, Any]]:
    """Parse and bulk-insert shelter records from CSV text with strict validation."""
    if not csv_text or not csv_text.strip():
        raise ValueError("Uploaded CSV file is empty.")

    f = io.StringIO(csv_text.strip())
    reader = csv.DictReader(f)
    if not reader.fieldnames:
        raise ValueError("Invalid CSV format: no header row found.")

    # Normalize header mapping
    fn_map = {fn.strip().lower(): fn for fn in reader.fieldnames if fn}
    req_fields = ["shelter_name", "location", "capacity", "occupancy"]
    for req in req_fields:
        if req not in fn_map:
            raise ValueError(f"Missing required CSV column '{req}'. Required columns: {', '.join(req_fields)}")

    imported_records = []
    row_num = 1
    for row in reader:
        row_num += 1
        name = str(row.get(fn_map["shelter_name"], "")).strip()
        location = str(row.get(fn_map["location"], "")).strip()
        raw_cap = row.get(fn_map["capacity"], "")
        raw_occ = row.get(fn_map["occupancy"], "")

        if not name or not location:
            raise ValueError(f"Row {row_num}: Shelter name and location cannot be empty.")

        try:
            capacity = int(float(raw_cap))
        except (ValueError, TypeError):
            raise ValueError(f"Row {row_num}: Capacity must be a valid positive integer. Received: '{raw_cap}'")
        if capacity <= 0:
            raise ValueError(f"Row {row_num}: Capacity must be greater than zero. Received: {capacity}")

        try:
            occupancy = int(float(raw_occ))
        except (ValueError, TypeError):
            raise ValueError(f"Row {row_num}: Occupancy must be a valid non-negative integer. Received: '{raw_occ}'")
        if occupancy < 0:
            raise ValueError(f"Row {row_num}: Occupancy cannot be negative. Received: {occupancy}")
        if occupancy > capacity:
            raise ValueError(f"Row {row_num}: Occupancy ({occupancy}) cannot exceed capacity ({capacity}).")

        shelter_id = None
        if "shelter_id" in fn_map and row.get(fn_map["shelter_id"]):
            shelter_id = str(row.get(fn_map["shelter_id"])).strip()

        notes = ""
        if "notes" in fn_map and row.get(fn_map["notes"]):
            notes = str(row.get(fn_map["notes"])).strip()

        record = create_shelter({
            "shelter_id": shelter_id,
            "shelter_name": name,
            "location": location,
            "capacity": capacity,
            "occupancy": occupancy,
            "data_source": USER_DATA_SOURCE,
            "notes": notes
        })
        imported_records.append(record)

    if not imported_records:
        raise ValueError("CSV contained no data rows.")
    return imported_records


def generate_resource_csv_template() -> str:
    """Return a standard downloadable CSV template for resource bulk imports."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["resource_id", "resource_type", "location", "required_quantity", "available_quantity", "unit", "notes"])
    writer.writerow(["RES-IMPORT-01", "Drinking Water", "San Juan", 12000, 8000, "Liters", "Bulk pallets staging"])
    writer.writerow(["RES-IMPORT-02", "Emergency Cots", "Metro Valley", 600, 600, "units", "Foldable metal cots"])
    return output.getvalue()


def import_resources_csv(csv_text: str) -> List[Dict[str, Any]]:
    """Parse and bulk-insert resource records from CSV text with strict validation."""
    if not csv_text or not csv_text.strip():
        raise ValueError("Uploaded CSV file is empty.")

    f = io.StringIO(csv_text.strip())
    reader = csv.DictReader(f)
    if not reader.fieldnames:
        raise ValueError("Invalid CSV format: no header row found.")

    fn_map = {fn.strip().lower(): fn for fn in reader.fieldnames if fn}
    req_fields = ["resource_type", "location", "required_quantity", "available_quantity", "unit"]
    for req in req_fields:
        if req not in fn_map:
            raise ValueError(f"Missing required CSV column '{req}'. Required columns: {', '.join(req_fields)}")

    imported_records = []
    row_num = 1
    for row in reader:
        row_num += 1
        res_type = str(row.get(fn_map["resource_type"], "")).strip()
        location = str(row.get(fn_map["location"], "")).strip()
        unit = str(row.get(fn_map["unit"], "")).strip()
        raw_req = row.get(fn_map["required_quantity"], "")
        raw_avail = row.get(fn_map["available_quantity"], "")

        if not res_type or not location or not unit:
            raise ValueError(f"Row {row_num}: Resource type, location, and unit cannot be empty.")

        try:
            req_qty = float(raw_req)
        except (ValueError, TypeError):
            raise ValueError(f"Row {row_num}: Required quantity must be a valid number. Received: '{raw_req}'")
        if req_qty < 0:
            raise ValueError(f"Row {row_num}: Required quantity cannot be negative. Received: {req_qty}")

        try:
            avail_qty = float(raw_avail)
        except (ValueError, TypeError):
            raise ValueError(f"Row {row_num}: Available quantity must be a valid number. Received: '{raw_avail}'")
        if avail_qty < 0:
            raise ValueError(f"Row {row_num}: Available quantity cannot be negative. Received: {avail_qty}")

        resource_id = None
        if "resource_id" in fn_map and row.get(fn_map["resource_id"]):
            resource_id = str(row.get(fn_map["resource_id"])).strip()

        notes = ""
        if "notes" in fn_map and row.get(fn_map["notes"]):
            notes = str(row.get(fn_map["notes"])).strip()

        record = create_resource({
            "resource_id": resource_id,
            "resource_type": res_type,
            "location": location,
            "required_quantity": req_qty,
            "available_quantity": avail_qty,
            "unit": unit,
            "data_source": USER_DATA_SOURCE,
            "notes": notes
        })
        imported_records.append(record)

    if not imported_records:
        raise ValueError("CSV contained no data rows.")
    return imported_records

