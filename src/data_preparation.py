"""Step 1: prepare extracted monthly features before any model selection.

Run from your project root:
    python src/data_preparation.py

Or supply paths explicitly:
    python data_preparation.py --data-dir "path/Feature data" --output-dir "path/artifacts"

This step does not fit a scaler, PCA, or a prediction model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "june": 6,
    "july": 7, "aug": 8, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
RAW_COLUMNS = [
    "id", "type", "taxi_time", "gate (block) hour", "runway hour", "rwy_day",
    "distance", "angle", "angle error", "shortest path", "distance_gate",
    "distance_long", "distance_else", "rwy", "rwy_num", "operation_mode",
    "other_moving_ac", "QDepDep", "QDepArr", "QArrDep", "QArrArr",
    "NDepDep", "NDepArr", "NArrDep", "NArrArr",
]
NUMERIC_COLUMNS = [column for column in RAW_COLUMNS if column not in {"id", "type", "rwy"}]
VEHICLE_PATTERN = (
    r"^(?:(?:OPS|LIGHT|SNOW|M\s*FIRE|FIRE|SWEEPER|SWEEP|GRASS|GRIP|GRIT|"
    r"FOLLOW\s*ME|RANGER|RESCUE|FUEL|TUG|AMBULANCE|POLICE)\s*\d+|ICE\s+\d+)(?:\s|$)"
)
# ICE is also Icelandair's ICAO operator code: compact ICE440 is a flight.
# The supplied ground-vehicle identifiers use a space, such as ICE 1.


def read_monthly_files(data_dir: Path) -> pd.DataFrame:
    """Read the known month files and preserve the source of every row."""
    frames = []
    for name, month in MONTHS.items():
        matches = [path for path in data_dir.glob("features_*.csv")
                   if path.stem.lower() == f"features_{name}"]
        if len(matches) != 1:
            raise ValueError(f"Expected one features_{name}.csv in {data_dir}; found {len(matches)}")
        path = matches[0]
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"id": "string", "type": "string", "rwy": "string"})
        if frame.columns.tolist() != RAW_COLUMNS:
            raise ValueError(f"Unexpected column schema in {path.name}")
        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["source_file"] = path.name
        frame["source_month"] = month
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def prepare_records(merged: pd.DataFrame, year: int = 2021, ambiguous_identifiers=None):
    """Return prepared, excluded, and review rows with reproducible counts.

    A record belongs to the file named for its ID date's month. This removes
    duplicated overlap days without randomly keeping a conflicting copy.
    The date is a source observation date, not a verified taxi-start timestamp.
    """
    frame = merged.copy()
    frame["id"] = frame["id"].str.strip()
    frame["type"] = frame["type"].str.strip().str.lower()
    date_text = frame["id"].str.extract(r"^(\d{2}-[A-Za-z]{3}-\d{4})_", expand=False)
    frame["source_date"] = pd.to_datetime(date_text, format="%d-%b-%Y", errors="coerce")
    frame["source_identifier"] = frame["id"].str.replace(r"^\d{2}-[A-Za-z]{3}-\d{4}_", "", regex=True)
    frame["exclusion_reason"] = ""

    def exclude(mask, reason):
        eligible = mask.fillna(False) & frame["exclusion_reason"].eq("")
        frame.loc[eligible, "exclusion_reason"] = reason

    exclude(frame["source_date"].isna(), "invalid_source_date")
    exclude(frame["source_date"].dt.year.ne(year), "outside_requested_year")
    exclude(frame["source_date"].dt.month.ne(frame["source_month"]), "month_boundary_overlap")
    exclude(~frame["type"].isin(["arrival", "departure"]), "invalid_movement_type")
    valid_target = np.isfinite(frame["taxi_time"]) & frame["taxi_time"].gt(0)
    exclude(~valid_target, "nonpositive_or_missing_taxi_time")
    known_vehicle = frame["source_identifier"].str.upper().str.contains(VEHICLE_PATTERN, regex=True, na=False)
    exclude(known_vehicle, "known_vehicle_identifier")
    if ambiguous_identifiers is not None:
        exclude(frame["id"].isin(ambiguous_identifiers), "ambiguous_aircraft_identity_for_daily_id")

    eligible = frame.loc[frame["exclusion_reason"].eq("")].copy()
    duplicate = eligible.duplicated(RAW_COLUMNS, keep="first")
    frame.loc[eligible.index[duplicate], "exclusion_reason"] = "exact_duplicate"
    prepared = frame.loc[frame["exclusion_reason"].eq("")].copy()
    if prepared.duplicated(["id", "type"], keep=False).any():
        raise ValueError("Conflicting id/type records remain after calendar selection; inspect them before continuing")

    # Direction-normalized start-side factors. These are derived columns,
    # rather than eight original features selected by PCA.
    departure = prepared["type"].eq("departure")
    prepared["start_hour"] = np.where(departure, prepared["gate (block) hour"], prepared["runway hour"])
    prepared["n_departing_at_start"] = np.where(departure, prepared["NDepDep"], prepared["NArrDep"])
    prepared["n_arriving_at_start"] = np.where(departure, prepared["NDepArr"], prepared["NArrArr"])

    prepared["review_reasons"] = ""
    def flag(mask, reason):
        prepared.loc[mask, "review_reasons"] += reason + ";"

    # Flags are review aids, not automatic declarations of invalid flights.
    # They depend partly on the target and must never enter model predictors.
    flag(prepared["taxi_time"].gt(120), "duration_over_120_minutes")
    flag(prepared["taxi_time"].lt(1), "duration_under_1_minute")
    flag(prepared["rwy"].isna() | prepared["rwy"].fillna("").str.strip().eq(""), "runway_name_missing")
    start_columns = ["start_hour", "n_departing_at_start", "n_arriving_at_start"]
    flag(~np.isfinite(prepared[start_columns]).all(axis=1), "start_factor_missing_or_nonfinite")
    flag(prepared["start_hour"].notna() & ~prepared["start_hour"].between(0, 23), "start_hour_outside_0_23")
    flag(prepared[["n_departing_at_start", "n_arriving_at_start"]].lt(0).any(axis=1), "negative_start_congestion")
    prepared["review_reasons"] = prepared["review_reasons"].str.rstrip(";")
    prepared = prepared.sort_values(["source_date", "id", "type"], kind="stable").reset_index(drop=True)
    excluded = frame.loc[frame["exclusion_reason"].ne("")].copy()
    review = prepared.loc[prepared["review_reasons"].ne("")].copy()
    summary = {
        "preparation_policy_version": 3 if ambiguous_identifiers is not None else 2,
        "input_rows": len(frame), "prepared_rows": len(prepared),
        "excluded_rows": len(excluded), "review_rows": len(review),
        "exclusion_counts": excluded["exclusion_reason"].value_counts().to_dict(),
        "movement_types": prepared["type"].value_counts().to_dict(),
        "source_year": year,
        "notes": [
            "No scaler, PCA or model has been fitted; no feature ranking has been performed",
            "Rows flagged for review remain in the prepared table; Step 2 records the modelling policy",
            "Unknown identifiers are retained; known vehicle syntax alone is excluded",
            "Metadata and review/exclusion flags must never be used as predictors",
            "Original supplied labels remain unverified observed-duration estimates",
            "Route distance/angle features are retrospective unless planned equivalents are available",
        ],
    }
    return prepared, excluded, review, summary


def save_preparation(output_dir, merged, prepared, excluded, review, summary):
    """Preserve an existing preparation; a policy change needs a new directory."""
    output_dir = Path(output_dir)
    content = {name: frame.to_csv(index=False, lineterminator="\n") for name, frame in [
        ("features_merged.csv", merged), ("features_prepared.csv", prepared),
        ("records_excluded.csv", excluded), ("records_for_review.csv", review),
    ]}
    content["preparation_summary.json"] = json.dumps(summary, indent=2, allow_nan=False)
    if any((output_dir / name).exists() for name in content):
        identical = all((output_dir / name).is_file() and
                        (output_dir / name).read_text(encoding="utf-8") == value
                        for name, value in content.items())
        if not identical:
            raise ValueError("Existing preparation differs; preserve it and use a new --output-dir, "
                             "for example artifacts/v2/step01. Do not overwrite a frozen benchmark.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in content.items():
        with (output_dir / name).open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)


def main():
    parser = argparse.ArgumentParser(description="Step 1: prepare monthly extracted taxi features")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "Feature data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "step01")
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--identity-audit", type=Path, help="Quarantine confirmed daily-ID aircraft conflicts from identity_audit.py")
    args = parser.parse_args()
    try:
        merged = read_monthly_files(args.data_dir)
        if args.identity_audit:
            from identity_audit import load_identity_audit
            from create_splits import sha256_file
            ambiguous = load_identity_audit(args.identity_audit, args.data_dir)
        else:
            ambiguous = None
        prepared, excluded, review, summary = prepare_records(merged, args.year, ambiguous)
        if args.identity_audit:
            summary["identity_audit_sha256"] = sha256_file(args.identity_audit)
        save_preparation(args.output_dir, merged, prepared, excluded, review, summary)
        print(f"Merged: {summary['input_rows']:,} rows")
        print(f"Prepared: {summary['prepared_rows']:,} rows")
        print(f"Excluded: {summary['excluded_rows']:,} rows")
        print(f"For review: {summary['review_rows']:,} rows (still included in prepared data)")
        print(f"Saved to {args.output_dir}")
    except (OSError, ValueError) as error:
        parser.exit(2, f"Data preparation failed: {error}\n")


if __name__ == "__main__":
    main()
