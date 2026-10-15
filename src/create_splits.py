"""Step 2: freeze an auditable retrospective taxi-time benchmark.

Run after data_preparation.py. No scaler, imputer, PCA or model is fitted here.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "taxi_time"
NUMERIC_FEATURES = [
    "start_hour", "distance", "angle", "shortest path", "distance_gate",
    "distance_long", "distance_else", "n_departing_at_start", "n_arriving_at_start",
]
FEATURES = ["type", *NUMERIC_FEATURES]
METADATA = ["id", "source_date"]
DISTANCES = ["distance", "shortest path", "distance_gate", "distance_long", "distance_else"]
COUNTS = ["n_departing_at_start", "n_arriving_at_start"]


def modelling_table(prepared: pd.DataFrame):
    """Validate Step 1 output and isolate the explicit candidate feature pool.

    Positive duration extremes and missing runway names remain eligible. Bad
    numeric predictor values become missing, with originals recorded separately.
    A later training-only imputer must handle missing values.
    """
    required = [*METADATA, TARGET, *FEATURES]
    missing = set(required).difference(prepared.columns)
    if missing:
        raise ValueError(f"Step 1 columns missing: {sorted(missing)}")
    frame = prepared.copy()
    if frame.empty:
        raise ValueError("Prepared dataset is empty")
    frame["source_date"] = pd.to_datetime(frame["source_date"], format="ISO8601", errors="coerce").dt.normalize()
    frame["id"] = frame["id"].astype("string").str.strip()
    frame["type"] = frame["type"].astype("string").str.strip().str.lower()
    if frame["source_date"].isna().any() or frame["id"].isna().any() or frame["id"].eq("").any():
        raise ValueError("Missing identifier or invalid source date; rerun Step 1")
    if not frame["type"].isin(["arrival", "departure"]).all():
        raise ValueError("Unexpected movement type; rerun Step 1")
    if frame.duplicated(["id", "type"]).any():
        raise ValueError("Duplicate id/type records; resolve them in Step 1")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    if not (np.isfinite(frame[TARGET]) & frame[TARGET].gt(0)).all():
        raise ValueError("Nonpositive or nonfinite target; rerun Step 1")

    changes = []
    for feature in NUMERIC_FEATURES:
        original = frame[feature].copy()
        values = pd.to_numeric(original, errors="coerce").astype(float)
        reasons = pd.Series("", index=frame.index, dtype="string")
        reasons.loc[original.notna() & values.isna()] = "unparseable_numeric"
        reasons.loc[values.notna() & ~np.isfinite(values)] = "nonfinite_numeric"
        if feature in DISTANCES:
            reasons.loc[np.isfinite(values) & values.lt(0)] = "negative_distance"
        elif feature == "start_hour":
            invalid = np.isfinite(values) & (~values.between(0, 23) | values.mod(1).ne(0))
            reasons.loc[invalid] = "hour_not_integer_0_to_23"
        elif feature in COUNTS:
            invalid = np.isfinite(values) & (values.lt(0) | values.mod(1).ne(0))
            reasons.loc[invalid] = "count_not_nonnegative_integer"
        mask = reasons.ne("")
        if mask.any():
            audit = frame.loc[mask, ["id", "type", "source_date"]].copy()
            audit["feature"] = feature
            audit["original_value"] = original.loc[mask]
            audit["reason"] = reasons.loc[mask]
            changes.append(audit)
        frame[feature] = values.mask(mask)

    # These are review flags, never predictors or grounds for automatic removal.
    flags = frame.get("review_reasons", pd.Series("", index=frame.index)).fillna("").astype(str)
    for mask, reason in [
        (frame["angle"].lt(0), "negative_angle_needs_definition_check"),
        (frame["shortest path"].eq(0), "zero_shortest_path_needs_route_check"),
        (frame[NUMERIC_FEATURES].isna().any(axis=1), "candidate_value_missing"),
    ]:
        flags.loc[mask] = flags.loc[mask].map(lambda value: f"{value};{reason}".strip(";"))
    provenance = frame[["id", "type", "source_date"]].copy()
    if "source_file" in frame:
        provenance["source_file"] = frame["source_file"]
    provenance["review_reasons"] = flags
    data = frame[[*METADATA, TARGET, *FEATURES]].copy()
    data = data.sort_values(["source_date", "id", "type"], kind="stable").reset_index(drop=True)
    adjustments = pd.concat(changes, ignore_index=True) if changes else pd.DataFrame(
        columns=["id", "type", "source_date", "feature", "original_value", "reason"]
    )
    return data, provenance, adjustments


def chronological_split(data: pd.DataFrame, gap_days: int = 1):
    """Choose whole-date cutoffs closest to 70% and 90% cumulative ROW counts.

    Only dates and counts determine the cutoffs. Calendar-day gaps follow the
    training and validation cutoffs. The source date is not an exact timestamp;
    this gap is a conservative precaution, not a verified session-overlap purge.
    """
    if gap_days < 0:
        raise ValueError("gap_days must be nonnegative")
    if data["source_date"].isna().any():
        raise ValueError("Source dates must be valid before splitting")
    dates = data.groupby("source_date", sort=True).size()
    if len(dates) < 3:
        raise ValueError("At least three distinct source dates are needed")
    cumulative = dates.cumsum().to_numpy()
    n = len(data)
    train_index = int(np.argmin(np.abs(cumulative[:-2] - 0.70 * n)))
    possible_validation = np.arange(train_index + 1, len(dates) - 1)
    validation_index = int(possible_validation[
        np.argmin(np.abs(cumulative[possible_validation] - 0.90 * n))
    ])
    train_end, validation_end = dates.index[train_index], dates.index[validation_index]
    training_gap_end = train_end + pd.Timedelta(days=gap_days)
    validation_gap_end = validation_end + pd.Timedelta(days=gap_days)
    assignment = pd.Series("boundary_gap", index=data.index, name="split")
    assignment.loc[data["source_date"].le(train_end)] = "train"
    assignment.loc[data["source_date"].gt(training_gap_end) &
                   data["source_date"].le(validation_end)] = "validation"
    assignment.loc[data["source_date"].gt(validation_gap_end)] = "test"
    partitions = {name: data.loc[assignment.eq(name)].copy()
                  for name in ["train", "validation", "test", "boundary_gap"]}
    if any(partitions[name].empty for name in ["train", "validation", "test"]):
        raise ValueError("A partition is empty after date gaps; use more dates or a smaller --gap-days")
    used_rows = sum(len(partitions[name]) for name in ["train", "validation", "test"])
    details = {}
    for name, part in partitions.items():
        details[name] = {
            "rows": len(part), "date_groups": int(part["source_date"].nunique()),
            "first_date": None if part.empty else part["source_date"].min().strftime("%Y-%m-%d"),
            "last_date": None if part.empty else part["source_date"].max().strftime("%Y-%m-%d"),
            "share_of_modelling_rows": len(part) / n,
            "share_of_used_rows": len(part) / used_rows if name != "boundary_gap" else None,
            "movement_types": {str(k): int(v) for k, v in part["type"].value_counts().items()},
        }
    return partitions, assignment, {
        "requested_row_fractions": {"train": 0.70, "validation": 0.20, "test": 0.10},
        "training_cutoff": train_end.strftime("%Y-%m-%d"),
        "validation_cutoff": validation_end.strftime("%Y-%m-%d"),
        "gap_days": gap_days, "partitions": details,
    }


def training_profile(train: pd.DataFrame):
    """Descriptive candidate statistics from training records only; no ranking."""
    rows = []
    for feature in NUMERIC_FEATURES:
        values = train[feature]
        row = {"feature": feature, "missing": int(values.isna().sum()),
               "unique_nonmissing": int(values.nunique()),
               "constant_in_training": bool(values.nunique() <= 1)}
        for name, value in {"min": values.min(), "median": values.median(),
                            "max": values.max(), "mean": values.mean(), "std": values.std()}.items():
            row[name] = float(value) if pd.notna(value) else None
        rows.append(row)
    return pd.DataFrame(rows)


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Step 2: retrospective modelling dataset and chronological split")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "artifacts" / "step01" / "features_prepared.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "step02")
    parser.add_argument("--gap-days", type=int, default=1)
    args = parser.parse_args()
    try:
        fingerprint = sha256_file(args.input)
        prepared = pd.read_csv(args.input, dtype={"id": "string", "type": "string", "source_file": "string"})
        data, provenance, adjustments = modelling_table(prepared)
        partitions, assignment, split_details = chronological_split(data, args.gap_days)
        manifest = {
            "protocol_version": 1, "prediction_task": "retrospective_completed_journey_benchmark",
            "target": TARGET, "target_units": "minutes", "candidate_features": FEATURES,
            "numeric_candidates": NUMERIC_FEATURES, "categorical_candidates": ["type"],
            "metadata_never_predictors": METADATA, "input_sha256": fingerprint,
            "modelling_rows": len(data), "numeric_adjustments": len(adjustments),
            **split_details,
            "policy": {
                "positive_duration_extremes": "retained; flags do not prove invalid labels",
                "missing_runway_name": "retained; runway name is outside this candidate pool",
                "zero_shortest_path": "retained and flagged; no verified invalid-value contract",
                "negative_angle": "retained and flagged; signed-angle semantics are unverified",
                "invalid_numeric_predictor": "record original; replace with missing; impute in training later",
                "distance_redundancy": "do not select total distance together with all three distance components",
                "scaling_pca_feature_selection": "fit on training only in the next step",
                "feature_subset_choice": "compare a small set of 7/8-factor alternatives on validation only",
                "test_set": "reserve for final evaluation after all choices are frozen",
                "chronology_limit": "whole source-date groups; exact taxi-start/end timestamps unavailable",
            },
        }
        manifest_path = args.output_dir / "split_manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != manifest:
                raise ValueError("Existing split manifest differs. Use a new --output-dir to preserve the old benchmark")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.output_dir / "modelling.csv", index=False)
        for name, part in partitions.items():
            part.to_csv(args.output_dir / f"{name}.csv", index=False)
        provenance = provenance.merge(data[["id", "type"]].assign(split=assignment.to_numpy()),
                                      on=["id", "type"], how="left", validate="one_to_one")
        provenance.to_csv(args.output_dir / "record_manifest.csv", index=False)
        provenance.loc[provenance["review_reasons"].ne("")].to_csv(args.output_dir / "review_records.csv", index=False)
        adjustments.to_csv(args.output_dir / "numeric_adjustments.csv", index=False)
        training_profile(partitions["train"]).to_csv(args.output_dir / "training_feature_profile.csv", index=False)
        flags = provenance["review_reasons"].str.get_dummies(sep=";").sum()
        (args.output_dir / "review_summary.json").write_text(json.dumps({
            "review_rows": int(provenance["review_reasons"].ne("").sum()),
            "flag_counts": {str(k): int(v) for k, v in flags.items()},
            "note": "Flags can overlap; all flagged records remain eligible except separately saved date gaps",
        }, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        for name in ["train", "validation", "test", "boundary_gap"]:
            details = split_details["partitions"][name]
            print(f"{name}: {details['rows']:,} rows, {details['first_date']} to {details['last_date']}")
        print(f"Candidate pool: {len(FEATURES)} factors; final 7/8 factors not selected yet")
        print(f"Saved to {args.output_dir}")
    except (OSError, ValueError) as error:
        parser.exit(2, f"Split preparation failed: {error}\n")


if __name__ == "__main__":
    main()
