"""Build a conservative feature cohort from the recorded source-session audit.

Only records with one supported raw-observation session under the explicit UTC
or Europe/London clock hypotheses are retained. This is an auditable screening
rule, not proof that the observations delimit an official taxi event.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_supported_cohort(prepared_path, validation_path, output_dir):
    prepared_path, validation_path = Path(prepared_path), Path(validation_path)
    output_dir = Path(output_dir)
    prepared = pd.read_csv(prepared_path, dtype={"id": "string", "type": "string"})
    audit = pd.read_csv(validation_path, dtype={"id": "string", "type": "string"})
    keys = ["id", "type"]
    for label, frame in [("prepared features", prepared), ("source validation", audit)]:
        if not set(keys).issubset(frame.columns) or frame[keys].isna().any().any():
            raise ValueError(f"{label} has missing record keys")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} contains duplicate id/type keys")
    if len(prepared) != len(audit):
        raise ValueError("Prepared features and source validation have different row counts")
    joined = prepared[keys].merge(audit[keys + ["status", "flags", "mapping_names",
        "raw_session_count", "source_address_count", "compatible_session_count",
        "snap_span_minutes"]], on=keys, how="outer", indicator=True, validate="one_to_one")
    if not joined["_merge"].eq("both").all():
        raise ValueError("Prepared features and source validation record keys do not match")
    supported_keys = joined.loc[joined.status.eq("supported_observation_session"), keys]
    supported = prepared.merge(supported_keys, on=keys, how="inner", validate="one_to_one")
    excluded = prepared.merge(joined.loc[~joined.status.eq("supported_observation_session")],
                              on=keys, how="inner", validate="one_to_one")
    if supported.empty:
        raise ValueError("No records meet the declared source-session support rule")
    supported = supported.sort_values(["source_date", "id", "type"], kind="stable")
    excluded = excluded.sort_values(["source_date", "id", "type"], kind="stable")
    summary = {
        "protocol_version": 1,
        "retained_status": "supported_observation_session",
        "input_rows": int(len(prepared)),
        "supported_rows": int(len(supported)),
        "excluded_for_source_review_rows": int(len(excluded)),
        "support_rule": "unique snap interval contained in one >=2-observation source session under UTC/London clock hypotheses, with no unresolved audit flags",
        "source_validation_sha256": sha256(validation_path),
        "prepared_features_sha256": sha256(prepared_path),
        "target_or_predictors_modified": False,
        "notes": [
            "This filters on source-session evidence, never duration cutoff or model error",
            "A supported observation session is not an official or independently verified taxi event",
            "The clock zone is unresolved; records are retained when exactly one session mapping is supported",
            "The December dates were previously inspected and remain development evaluation",
        ],
    }
    payloads = {
        "features_supported.csv": supported.to_csv(index=False, lineterminator="\n"),
        "features_for_source_review.csv": excluded.to_csv(index=False, lineterminator="\n"),
        "cohort_summary.json": json.dumps(summary, indent=2, allow_nan=False) + "\n",
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Cohort output directory is nonempty; preserve it and choose a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        (output_dir / name).write_text(content, encoding="utf-8", newline="\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = build_supported_cohort(args.prepared, args.validation, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Cohort construction failed: {error}\n")
    print(f"Source-supported cohort: {summary['supported_rows']:,}/{summary['input_rows']:,}; "
          f"flagged for source review: {summary['excluded_for_source_review_rows']:,}")


if __name__ == "__main__":
    main()
