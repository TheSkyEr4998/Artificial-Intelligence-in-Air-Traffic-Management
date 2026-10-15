"""Step 6: inspect the frozen benchmark without fitting or changing its scores.

Snapped data are diagnostics, not authoritative airport event labels. Long
interval/gap flags require review; they never automatically remove records.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from create_splits import sha256_file
from data_preparation import prepare_records, read_monthly_files
from evaluate import evaluate_models
from prediction_quality import training_support_profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def interval_diagnostics(frame):
    """Union coverage avoids double-counting overlapping snapped edge intervals."""
    times = frame[["t1", "t2"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(times).all(axis=1) & (times[:, 1] >= times[:, 0])
    intervals = times[valid]
    result = {"snapped_rows": len(frame), "invalid_interval_rows": int((~valid).sum()),
              "snap_index_groups": int(frame["index"].nunique())}
    if not len(intervals):
        return {**result, "span_minutes": None, "union_interval_minutes": None,
                "longest_edge_minutes": None, "largest_uncovered_gap_minutes": None,
                "first_timestamp_utc": None, "last_timestamp_utc": None}
    intervals = intervals[np.argsort(intervals[:, 0], kind="stable")]
    start, end = intervals[0]
    union_seconds = largest_gap = 0.0
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            union_seconds += end - start
            largest_gap = max(largest_gap, next_start - end)
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    union_seconds += end - start
    first, last = float(intervals[:, 0].min()), float(intervals[:, 1].max())
    return {**result, "span_minutes": (last-first)/60, "union_interval_minutes": union_seconds/60,
            "longest_edge_minutes": float(np.max(intervals[:, 1]-intervals[:, 0])/60),
            "largest_uncovered_gap_minutes": largest_gap/60,
            "first_timestamp_utc": pd.Timestamp(first, unit="s", tz="UTC").isoformat(),
            "last_timestamp_utc": pd.Timestamp(last, unit="s", tz="UTC").isoformat()}


def audit_snapped(prepared, snapped_dir):
    frames, fingerprints = [], {}
    for month in range(1, 13):
        path = Path(snapped_dir) / f"snapped{month}.csv"
        if not path.is_file():
            raise ValueError(f"Missing snapped input: {path}")
        frame = pd.read_csv(path)
        if frame.columns.tolist() != ["index", "flight_id", "edge_id", "t1", "t2"]:
            raise ValueError(f"Unexpected snapped schema: {path.name}")
        eligible = prepared.loc[pd.to_datetime(prepared.source_date).dt.month.eq(month)]
        identifiers = set(eligible.id)
        details = [dict(id=identifier, **interval_diagnostics(group))
                   for identifier, group in frame.loc[frame.flight_id.isin(identifiers)].groupby("flight_id", sort=False)]
        part = eligible[["id", "type", "source_date", "taxi_time"]].merge(
            pd.DataFrame(details, columns=["id", "snapped_rows", "invalid_interval_rows", "snap_index_groups",
                "span_minutes", "union_interval_minutes", "longest_edge_minutes", "largest_uncovered_gap_minutes",
                "first_timestamp_utc", "last_timestamp_utc"]), on="id", how="left", validate="many_to_one")
        # Both directions can share a daily ID; then these diagnostics cover the
        # whole ID, not a uniquely resolved feature-record event.
        part["feature_records_for_id"] = part.id.map(eligible.groupby("id").size())
        part["review_flags"] = ""
        for mask, reason in [
            (part.snapped_rows.isna(), "no_snapped_match"),
            (part.feature_records_for_id.gt(1), "multiple_feature_events_for_daily_id"),
            (part.snap_index_groups.gt(1), "multiple_snap_index_groups"),
            (part.invalid_interval_rows.gt(0), "invalid_snap_intervals"),
            (part.longest_edge_minutes.gt(15), "edge_interval_over_15_minutes"),
            (part.largest_uncovered_gap_minutes.gt(15), "uncovered_gap_over_15_minutes"),
        ]:
            part.loc[mask, "review_flags"] += reason + ";"
        part["review_flags"] = part.review_flags.str.rstrip(";")
        part["label_minus_daily_span_minutes"] = part.taxi_time - part.span_minutes
        frames.append(part)
        fingerprints[path.name] = sha256_file(path)
    return pd.concat(frames, ignore_index=True), fingerprints


def saved_error_diagnostics(predictions):
    rows = []
    for family, group in predictions.groupby("family", sort=False):
        actual = group.actual_minutes.to_numpy(dtype=float)
        error = group.predicted_minutes.to_numpy(dtype=float) - actual
        absolute, squared = np.abs(error), error**2
        for cutoff in [60, 120, 600]:
            mask = actual > cutoff
            rows.append({"family": family, "duration_over_minutes": cutoff, "rows": int(mask.sum()),
                         "fraction_of_records": float(mask.mean()),
                         "fraction_of_absolute_error": float(absolute[mask].sum()/absolute.sum()) if absolute.sum() else 0,
                         "fraction_of_squared_error": float(squared[mask].sum()/squared.sum()) if squared.sum() else 0})
    return pd.DataFrame(rows)


def development_baselines(train, validation):
    """Additional simple baselines: development evidence only, never new test scores."""
    pooled = float(train.taxi_time.median())
    type_medians = train.groupby("type").taxi_time.median()
    settings = {"pooled_training_median": np.full(len(validation), pooled),
                "movement_type_training_median": validation.type.map(type_medians).fillna(pooled).to_numpy()}
    rows = []
    for name, predicted in settings.items():
        error = predicted - validation.taxi_time.to_numpy()
        rows.append({"baseline": name, "validation_rows": len(validation),
                     "mae_minutes": float(np.abs(error).mean()), "rmse_minutes": float(np.sqrt(np.mean(error**2)))})
    return {"fit_partition": "training only", "scored_partition": "validation only",
            "test_scored": False, "type_medians_minutes": {k: float(v) for k, v in type_medians.items()},
            "results": rows, "note": "Added after the original test was observed; exploratory development evidence"}


def run_audit(project_dir, output_dir, artifacts_dir=None, feature_dir=None, snapped_dir=None, prepared_path=None):
    project_dir, output_dir = Path(project_dir), Path(output_dir)
    artifacts = Path(artifacts_dir) if artifacts_dir else project_dir / "artifacts"
    feature_dir = Path(feature_dir) if feature_dir else project_dir / "Feature data"
    snapped_dir = Path(snapped_dir) if snapped_dir else project_dir / "Snapped data"
    # Require a sealed prior evaluation so this command cannot perform a first
    # test evaluation or silently fit anything. evaluate_models verifies/cache-reuses it.
    if not (artifacts / "step05" / "evaluation_manifest.json").is_file():
        raise ValueError("Complete the original frozen Step 5 evaluation before this audit")
    result = evaluate_models(artifacts / "step02", artifacts / "step03", artifacts / "step04",
                             artifacts / "step05", prepared_path=prepared_path)
    predictions = pd.read_csv(artifacts / "step05" / "test_predictions.csv")
    # Recompute stored errors independently: no model inference or test selection.
    for family, group in predictions.groupby("family"):
        error = group.predicted_minutes.to_numpy() - group.actual_minutes.to_numpy()
        for name, value in [("mae_minutes", np.abs(error).mean()), ("rmse_minutes", np.sqrt(np.mean(error**2)))]:
            if not np.isclose(value, result["metrics"][family][name], rtol=1e-10, atol=1e-8):
                raise ValueError(f"Saved metric/prediction disagreement: {family}/{name}")
    prepared = pd.read_csv(artifacts / "step01" / "features_prepared.csv")
    exclusions = pd.read_csv(artifacts / "step01" / "records_excluded.csv")
    preparation = json.loads((artifacts / "step01" / "preparation_summary.json").read_text(encoding="utf-8"))
    quarantine = exclusions.exclusion_reason.eq("ambiguous_aircraft_identity_for_daily_id")
    identifiers = set(exclusions.loc[quarantine, "id"]) if preparation.get("preparation_policy_version") == 3 else None
    corrected, _, _, correction = prepare_records(read_monthly_files(feature_dir), ambiguous_identifiers=identifiers)
    restored = corrected.merge(prepared[["id", "type"]], how="left", on=["id", "type"], indicator=True)
    restored = restored.loc[restored._merge.eq("left_only")].drop(columns="_merge")
    removed = prepared.merge(corrected[["id", "type"]], how="left", on=["id", "type"], indicator=True)
    removed = removed.loc[removed._merge.eq("left_only")].drop(columns="_merge")
    snaps, snap_sha = audit_snapped(prepared, snapped_dir)
    train = pd.read_csv(artifacts / "step02" / "train.csv")
    validation = pd.read_csv(artifacts / "step02" / "validation.csv")
    selection = json.loads((artifacts / "step04" / "model_selection.json").read_text(encoding="utf-8"))
    if sha256_file(artifacts / "step02" / "train.csv") != selection["train_sha256"]:
        raise ValueError("Training data changed; cannot build a support profile")
    support = training_support_profile(train, selection["selected_factors"], selection["train_sha256"])
    tails = saved_error_diagnostics(predictions)
    merged = predictions.merge(snaps, on=["id", "type"], how="left", validate="many_to_one", suffixes=("", "_audit"))
    top = merged.loc[merged.family.eq(selection["winner_family"])].copy()
    top["absolute_error_minutes"] = (top.predicted_minutes-top.actual_minutes).abs()
    top = top.nlargest(20, "absolute_error_minutes")
    summary = {"audit_version": 1, "frozen_evaluation_verified": True, "stored_metrics_verified": True,
               "models_refitted": False, "frozen_results_changed": False,
               "original_prepared_rows": len(prepared), "corrected_prepared_rows": correction["prepared_rows"],
               "restored_aircraft_records": len(restored), "newly_removed_records": len(removed),
               "identity_conflict_records_quarantined": int(quarantine.sum()),
               "snapped_diagnostics_rows": len(snaps), "records_flagged_for_review": int(snaps.review_flags.ne("").sum()),
               "edge_interval_over_15_minutes": int(snaps.longest_edge_minutes.gt(15).sum()),
               "uncovered_gap_over_15_minutes": int(snaps.largest_uncovered_gap_minutes.gt(15).sum()),
               "input_sha256": {"step05_manifest": sha256_file(artifacts / "step05" / "evaluation_manifest.json"),
                                **snap_sha},
               "notes": ["Interval/gap thresholds are review aids, not label correction or automatic exclusions",
                         "A daily callsign can combine separate events; union coverage is not verified taxi time",
                         "Snapping can already bridge a data gap, so high union coverage is not proof of continuous observation",
                         "No target-duration filtering, prediction clipping, retraining, or new test selection was performed",
                         "The observed December test cannot serve as untouched evidence for subsequent tuning",
                         "Recorded scores always describe their frozen preparation policy; the original pre-audit run excluded 22 ICE airline records",
                         "Identity quarantine does not verify the event timing of retained records"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("restored_aircraft_records.csv", restored), ("newly_removed_records.csv", removed),
                        ("snapped_label_diagnostics.csv", snaps), ("saved_error_contributions.csv", tails),
                        ("largest_winner_errors.csv", top)]:
        frame.to_csv(output_dir / name, index=False)
    for name, value in [("audit_summary.json", summary), ("training_support.json", support),
                        ("development_baselines.json", development_baselines(train, validation))]:
        (output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Frozen results verified; restored airline records: {len(restored)}")
    print(f"Snapped records flagged for review: {summary['records_flagged_for_review']:,}; saved to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Audit saved results and snapped labels without changing the benchmark")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "step06")
    parser.add_argument("--artifacts-dir", type=Path, help="Use a separate revision artifacts directory")
    parser.add_argument("--feature-dir", type=Path, help="Monthly extracted feature files, if outside the project root")
    parser.add_argument("--snapped-dir", type=Path, help="Snapped source files, if outside the project root")
    parser.add_argument("--prepared-input", type=Path, help="Exact Step 2 input file when the run used a screened feature cohort")
    args = parser.parse_args()
    try:
        run_audit(args.project_dir, args.output_dir, args.artifacts_dir, args.feature_dir,
                  args.snapped_dir, args.prepared_input)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Quality audit failed: {error}\n")


if __name__ == "__main__":
    main()
