"""Audit source daily IDs for multiple aircraft addresses, without using targets.

Only syntactically valid six-hex-digit Mode S addresses count as aircraft
identities. Two distinct addresses make a daily ID ambiguous for a single-aircraft
journey label. Missing addresses are recorded, not treated as confirmed conflicts.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from create_splits import sha256_file
from data_preparation import MONTHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalize_addresses(series):
    values = series.astype("string").str.strip().str.lower().str.replace(r"^0x", "", regex=True)
    return values.where(values.str.fullmatch(r"[0-9a-f]{6}", na=False))


def audit_identity(raw_dir, feature_dir, output_dir, chunksize=250_000):
    raw_dir, feature_dir, output_dir = map(Path, [raw_dir, feature_dir, output_dir])
    pairs, sources, features = [], {}, {}
    total_rows = missing_rows = 0
    for month in range(1, 13):
        path = raw_dir / f"cleanedx_{month:02}.csv"
        if not path.is_file():
            raise ValueError(f"Missing raw cleaned file: {path}")
        digest = sha256_file(path)
        rows = 0
        for chunk in pd.read_csv(path, usecols=["my_id", "mode_s"], dtype="string", chunksize=chunksize):
            rows += len(chunk)
            chunk["my_id"] = chunk.my_id.str.strip()
            chunk["mode_s"] = normalize_addresses(chunk.mode_s)
            missing_rows += int(chunk.mode_s.isna().sum())
            distinct = chunk.dropna(subset=["my_id", "mode_s"]).drop_duplicates()
            date = pd.to_datetime(distinct.my_id.str.extract(r"^(\d{2}-[A-Za-z]{3}-\d{4})_", expand=False),
                                  format="%d-%b-%Y", errors="coerce")
            pairs.append(distinct.loc[date.dt.month.eq(month)])
        sources[path.name] = {"sha256": digest, "rows": rows}
        total_rows += rows
        print(f"Identity audit: month {month:02}; {rows:,} observations", flush=True)
    # Keep only the calendar owner's records, matching feature preparation.
    pair_table = pd.concat(pairs, ignore_index=True).drop_duplicates()
    counts = pair_table.groupby("my_id").mode_s.nunique()
    ambiguous = sorted(counts.index[counts.gt(1)].tolist())
    evidence = pair_table.loc[pair_table.my_id.isin(ambiguous)].sort_values(["my_id", "mode_s"])
    for name in MONTHS:
        matches = [path for path in feature_dir.glob("features_*.csv") if path.stem.lower() == f"features_{name}"]
        if len(matches) != 1:
            raise ValueError(f"Expected one features_{name}.csv")
        features[matches[0].name] = sha256_file(matches[0])
    summary = {"identity_audit_version": 1, "raw_observation_rows": total_rows,
               "rows_without_valid_mode_s": missing_rows, "daily_ids_with_valid_address": len(counts),
               "ambiguous_daily_ids": ambiguous, "ambiguous_daily_id_count": len(ambiguous),
               "source_files": sources, "feature_files_sha256": features,
               "rule": "At least two distinct valid Mode S addresses observed for the same daily ID",
               "target_used_for_rule": False,
               "notes": ["Quarantine is based on aircraft-identity ambiguity, not a taxi-duration cutoff",
                         "This conservative rule can quarantine valid individual events whose daily ID is reused; it does not prove every label is wrong",
                         "One address does not prove a continuous or correctly anchored taxi journey",
                         "Missing or malformed addresses do not establish an identity conflict",
                         "Original snapped routes and labels are not repaired by this audit"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence.rename(columns={"my_id": "id", "mode_s": "aircraft_address"}).to_csv(output_dir / "identity_conflicts.csv", index=False)
    (output_dir / "identity_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Confirmed ambiguous daily IDs: {len(ambiguous):,}; saved to {output_dir}")
    return summary


def load_identity_audit(path, feature_dir):
    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    if audit.get("identity_audit_version") != 1 or audit.get("target_used_for_rule") is not False:
        raise ValueError("Unexpected source identity audit protocol")
    identifiers = audit["ambiguous_daily_ids"]
    if not isinstance(identifiers, list) or not all(isinstance(value, str) for value in identifiers):
        raise ValueError("Invalid identity conflict list")
    if len(identifiers) != len(set(identifiers)) or len(identifiers) != audit["ambiguous_daily_id_count"]:
        raise ValueError("Identity conflict count/list disagree")
    for name, digest in audit["feature_files_sha256"].items():
        if Path(name).name != name or sha256_file(Path(feature_dir) / name) != digest:
            raise ValueError(f"Feature input changed since identity audit: {name}")
    if len(audit["feature_files_sha256"]) != 12:
        raise ValueError("Identity audit must cover all twelve feature files")
    return set(identifiers)


def main():
    parser = argparse.ArgumentParser(description="Audit all cleaned observations for daily-ID aircraft conflicts")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "Cleaned data")
    parser.add_argument("--feature-dir", type=Path, default=PROJECT_ROOT / "Feature data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "identity_audit")
    args = parser.parse_args()
    try:
        audit_identity(args.raw_dir, args.feature_dir, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Identity audit failed: {error}\n")


if __name__ == "__main__":
    main()
