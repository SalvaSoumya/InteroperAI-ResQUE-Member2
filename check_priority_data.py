import pandas as pd
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Change this if your 10,000-row CSV has a different filename.
DATASET_PATH = Path("job-1791397398354_processed.csv")

OUTPUT_DIR = Path("member2_ml_artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "priority_diagnostic_report.txt"

TARGET = "priority_level"

NUMERIC_CANDIDATES = [
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

CATEGORICAL_CANDIDATES = [
    "disaster_type",
    "incident_location",
    "infrastructure_damage",
    "response_status",
    "responding_agency",
    "required_resources",
    "available_resources",
    "incident_description",
]


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 75)
print("DISASTER DATASET - PRIORITY LABEL DIAGNOSTICS")
print("=" * 75)

if not DATASET_PATH.exists():
    print("\nERROR: Dataset file not found!")
    print("Expected location:", DATASET_PATH.resolve())
    print("Update DATASET_PATH to the correct CSV filename.")
    raise SystemExit(1)

try:
    df = pd.read_csv(DATASET_PATH)
except Exception as exc:
    print(f"ERROR: Could not read CSV: {exc}")
    raise SystemExit(1)

if df.empty:
    print("ERROR: Dataset is empty.")
    raise SystemExit(1)

print("\nDataset path:", DATASET_PATH.resolve())
print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nAvailable columns:")
print(df.columns.tolist())

if TARGET not in df.columns:
    print(f"\nERROR: '{TARGET}' column not found.")
    raise SystemExit(1)

# Capture report output.
report_lines = []


def log(message=""):
    """Print output and save it for the diagnostic report."""
    message = str(message)
    print(message)
    report_lines.append(message)


log("DISASTER DATASET PRIORITY DIAGNOSTICS")
log("=" * 65)
log(f"Dataset: {DATASET_PATH.resolve()}")
log(f"Dataset shape: {df.shape}")


# ============================================================
# 3. DATA QUALITY CHECK
# ============================================================

log("\n" + "=" * 65)
log("1. DATA QUALITY CHECK")
log("=" * 65)

missing_total = int(df.isna().sum().sum())
duplicate_total = int(df.duplicated().sum())

log(f"Total rows: {len(df)}")
log(f"Total columns: {len(df.columns)}")
log(f"Missing cells: {missing_total}")
log(f"Duplicate rows: {duplicate_total}")

log("\nMissing values by column:")
missing_by_column = df.isna().sum()
missing_by_column = missing_by_column[missing_by_column > 0]

log(
    missing_by_column.to_string()
    if not missing_by_column.empty
    else "No missing values found."
)

log("\nColumn data types:")
log(df.dtypes.to_string())


# ============================================================
# 4. PRIORITY LABEL COUNTS AND DISTRIBUTION
# ============================================================

log("\n" + "=" * 65)
log("2. PRIORITY LABEL DISTRIBUTION")
log("=" * 65)

priority_counts = df[TARGET].value_counts(dropna=False)

priority_percentages = (
    df[TARGET].value_counts(normalize=True, dropna=False) * 100
).round(2)

log("\nPriority counts:")
log(priority_counts.to_string())

log("\nPriority percentages:")
log(priority_percentages.to_string())

log(f"\nNumber of valid priority classes: {df[TARGET].nunique()}")
log(f"Missing priority labels: {int(df[TARGET].isna().sum())}")

if df[TARGET].nunique() < 2:
    log("WARNING: At least two valid classes are required for classification.")

valid_counts = df[TARGET].value_counts()

if len(valid_counts) > 1:
    imbalance_ratio = valid_counts.max() / valid_counts.min()
    log(f"Largest-to-smallest class ratio: {imbalance_ratio:.2f}")

    if imbalance_ratio > 3:
        log("WARNING: The class distribution may be imbalanced.")
    else:
        log("No severe class imbalance detected by this ratio check.")


# ============================================================
# 5. NUMERIC FEATURES BY PRIORITY
# ============================================================

log("\n" + "=" * 65)
log("3. NUMERIC FEATURES BY PRIORITY")
log("=" * 65)

numeric_cols = [
    col for col in NUMERIC_CANDIDATES
    if col in df.columns
]

# Convert numeric-looking values into numbers.
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if numeric_cols:
    grouped = df.groupby(TARGET, dropna=False)[numeric_cols]

    log("\nMEAN:")
    log(grouped.mean().round(2).to_string())

    log("\nMEDIAN:")
    log(grouped.median().round(2).to_string())

    log("\nSTANDARD DEVIATION:")
    log(grouped.std().round(2).to_string())

    log("\nMINIMUM:")
    log(grouped.min().round(2).to_string())

    log("\nMAXIMUM:")
    log(grouped.max().round(2).to_string())

    # Compare class means, but do not treat this as proof of
    # whether labels are correct or random.
    means = df.groupby(TARGET)[numeric_cols].mean()

    ranges = pd.DataFrame({
        "minimum_class_mean": means.min(),
        "maximum_class_mean": means.max(),
        "difference": means.max() - means.min(),
        "overall_mean": df[numeric_cols].mean(),
    })

    ranges["difference_as_pct_of_mean"] = (
        ranges["difference"]
        / ranges["overall_mean"].abs().replace(0, float("nan"))
        * 100
    )

    log("\nRANGE OF CLASS MEANS:")
    log(ranges.round(2).to_string())

    log(
        "\nNOTE: Similar class means may indicate weak separation. "
        "Inspect the full distributions before drawing conclusions."
    )
else:
    log("No matching numeric features found.")


# ============================================================
# 6. CATEGORICAL FEATURES BY PRIORITY
# ============================================================

log("\n" + "=" * 65)
log("4. CATEGORICAL FEATURES BY PRIORITY")
log("=" * 65)

categorical_cols = [
    col for col in CATEGORICAL_CANDIDATES
    if col in df.columns
]

for col in categorical_cols:
    log(f"\n--- {col.upper()} ---")

    category_table = pd.crosstab(
        df[col].fillna("MISSING"),
        df[TARGET].fillna("MISSING"),
        normalize="index"
    ) * 100

    log("Priority distribution within each category (%):")
    log(category_table.round(2).to_string())

    log("\nCategory counts:")
    log(
        df[col]
        .fillna("MISSING")
        .value_counts()
        .head(20)
        .to_string()
    )


# ============================================================
# 7. INFRASTRUCTURE DAMAGE ANALYSIS
# ============================================================

log("\n" + "=" * 65)
log("5. INFRASTRUCTURE DAMAGE BY PRIORITY")
log("=" * 65)

if "infrastructure_damage" in df.columns:
    damage_table = pd.crosstab(
        df["infrastructure_damage"].fillna("MISSING"),
        df[TARGET].fillna("MISSING"),
        normalize="index"
    ) * 100

    log(damage_table.round(2).to_string())
else:
    log("'infrastructure_damage' column not found.")


# ============================================================
# 8. NUMERIC CORRELATIONS
# ============================================================

log("\n" + "=" * 65)
log("6. NUMERIC CORRELATIONS")
log("=" * 65)

if len(numeric_cols) >= 2:
    correlation = df[numeric_cols].corr(numeric_only=True)
    log(correlation.round(3).to_string())
else:
    log("Not enough numeric features for correlation analysis.")

log(
    "\nNOTE: Correlations between input features do not establish "
    "whether priority labels are correct."
)


# ============================================================
# 9. POSSIBLE TARGET LEAKAGE
# ============================================================

log("\n" + "=" * 65)
log("7. POSSIBLE TARGET LEAKAGE CHECK")
log("=" * 65)

possible_leakage = [
    col for col in df.columns
    if any(
        term in col.lower()
        for term in [
            "priority_score",
            "risk_score",
            "severity_label",
            "priority_prediction",
            "predicted_priority",
        ]
    )
]

if possible_leakage:
    log("Review these potentially related columns:")
    log(possible_leakage)
    log(
        "Exclude fields that directly reveal the target or contain "
        "information unavailable at prediction time."
    )
else:
    log("No obvious target-related column names detected.")

log(
    "\nAlso verify that no input feature was created from priority_level "
    "or from information recorded after the incident outcome."
)


# ============================================================
# 10. SAMPLE RECORDS
# ============================================================

log("\n" + "=" * 65)
log("8. FIRST 10 RECORDS")
log("=" * 65)

sample_cols = [
    col for col in [
        "disaster_type",
        "priority_level",
        "affected_population",
        "reported_casualties",
        "injured_population",
        "displaced_population",
        "infrastructure_damage",
        "estimated_loss",
        "incident_description",
    ]
    if col in df.columns
]

log(df[sample_cols].head(10).to_string(index=False))


# ============================================================
# 11. FINAL SUMMARY AND RECOMMENDATIONS
# ============================================================

log("\n" + "=" * 65)
log("9. FINAL SUMMARY")
log("=" * 65)

log(f"Rows: {len(df)}")
log(f"Columns: {len(df.columns)}")
log(f"Missing cells: {missing_total}")
log(f"Duplicate rows: {duplicate_total}")
log(f"Priority classes: {df[TARGET].nunique()}")

log("\nRECOMMENDATIONS:")
log("1. Confirm how priority_level labels were created.")
log("2. Verify labels against trusted historical data or documented rules.")
log("3. Inspect numeric distributions and categorical relationships.")
log("4. Exclude target leakage and post-outcome information.")
log("5. Evaluate models on a held-out test set.")
log("6. Compare macro F1-score and balanced accuracy, not just accuracy.")
log("7. Do not claim reliable predictions until labels and evaluation are validated.")

with REPORT_PATH.open("w", encoding="utf-8") as report_file:
    report_file.write("\n".join(report_lines))

print("\n" + "=" * 75)
print("DIAGNOSTICS COMPLETED")
print("Report saved to:", REPORT_PATH.resolve())
print("=" * 75)