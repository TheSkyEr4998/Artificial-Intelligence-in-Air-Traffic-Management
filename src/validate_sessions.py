"""Trace recorded intervals to source observation sessions without changing labels.

Historical cleaned clocks lack a declared zone. UTC-clock and London-clock
interpretations are tested as hypotheses; neither is silently made ground truth.
The 2400-second gap comes from the supplied snap program's divide_flights rule.
This applies that observation-session rule independently of callsign presence.
It does not certify uninterrupted taxiing or choose a target-duration cutoff.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAP_SECONDS = 2400
BOUNDARY_TOLERANCE_SECONDS = 1  # Cleaned clocks have whole-second resolution.
POLICY = {
    "version": 1,
    "session_gap_seconds": GAP_SECONDS,
    "gap_provenance": "supplied snap.exe data.py divide_flights original lines 408-409",
    "gap_applies_to_all_callsigns": True,
    "clock_coordinates": "naive recorded clock; no historical timezone asserted",
    "mapping_hypotheses": ["UTC_clock", "Europe_London_clock"],
    "boundary_tolerance_seconds": BOUNDARY_TOLERANCE_SECONDS,
    "label_span_tolerance_minutes": 1e-8,
    "model_errors_used": False,
    "target_duration_cutoff": None,
    "labels_or_features_rewritten": False,
    "meaning_of_supported": "unique observation-session support under stated hypotheses; not verified airport events",
}
OUTPUTS = {"record_validation.csv", "source_sessions.csv", "session_validation_summary.json"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_clock_sessions(clock_seconds):
    """Return sorted unique clock ranges; equality at 2400 seconds stays together."""
    times = np.unique(np.asarray(clock_seconds, dtype=np.int64))
    if not len(times):
        return []
    divisions = np.flatnonzero(np.diff(times) > GAP_SECONDS) + 1
    return [(int(part[0]), int(part[-1]), len(part)) for part in np.split(times, divisions)]


def clock_hypotheses(first_epoch, last_epoch):
    """Compare inert epoch values with recorded clock strings, preserving DST flags."""
    first = pd.Timestamp(first_epoch, unit="s", tz="UTC")
    last = pd.Timestamp(last_epoch, unit="s", tz="UTC")
    result = []
    for name, zone in [("UTC_clock", "UTC"), ("Europe_London_clock", "Europe/London")]:
        start, end = first.tz_convert(zone), last.tz_convert(zone)
        result.append({"mapping": name,
                       "start_clock_seconds": start.tz_localize(None).value / 1e9,
                       "end_clock_seconds": end.tz_localize(None).value / 1e9,
                       "offset_changes": start.utcoffset() != end.utcoffset()})
    return result


def classify_interval(label_minutes, raw, snapped):
    """Classify provenance/continuity, with no access to fitted models or errors."""
    flags = []
    sessions = split_clock_sessions(raw.get("times", []))
    addresses = sorted(raw.get("addresses", set()))
    if not sessions:
        flags.append("no_source_observations")
    if raw.get("invalid_clocks", 0):
        flags.append("invalid_source_clock")
    if not addresses:
        flags.append("no_valid_source_aircraft_address")
    if len(addresses) > 1:
        flags.append("multiple_aircraft_addresses")
    if snapped is None:
        flags.append("no_snapped_match")
        return {"status": "needs_source_review", "flags": ";".join(flags), "raw_session_count": len(sessions),
                "source_address_count": len(addresses), "mapping_names": "", "compatible_session_count": 0,
                "snap_span_minutes": None, "first_snap_epoch": None, "last_snap_epoch": None}, sessions
    times = snapped[["t1", "t2"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(times).all(axis=1) & (times[:, 1] >= times[:, 0])
    if not valid.all():
        flags.append("invalid_snapped_interval")
    if snapped["index"].nunique() != 1:
        flags.append("multiple_snapped_index_groups")
    if not valid.any():
        return {"status": "needs_source_review", "flags": ";".join(flags), "raw_session_count": len(sessions),
                "source_address_count": len(addresses), "mapping_names": "", "compatible_session_count": 0,
                "snap_span_minutes": None, "first_snap_epoch": None, "last_snap_epoch": None}, sessions
    first, last = float(times[valid, 0].min()), float(times[valid, 1].max())
    span = (last - first) / 60
    if not np.isfinite(label_minutes) or not np.isclose(label_minutes, span, atol=1e-8, rtol=0):
        flags.append("feature_event_bounds_unresolved")
    matches, supported_names = set(), []
    overall_support = False
    for hypothesis in clock_hypotheses(first, last):
        start, end = hypothesis["start_clock_seconds"], hypothesis["end_clock_seconds"]
        within_source = bool(sessions and start <= end and start >= sessions[0][0] - 1 and end <= sessions[-1][1] + 1)
        if within_source and hypothesis["offset_changes"]:
            flags.append("clock_offset_transition_possible")
        overall_support |= within_source
        found = False
        for index, (raw_start, raw_end, count) in enumerate(sessions):
            if count >= 2 and start <= end and raw_start - 1 <= start and end <= raw_end + 1:
                matches.add((index, round(start, 6), round(end, 6)))
                found = True
        if found:
            supported_names.append(hypothesis["mapping"])
    if not matches:
        flags.append("observation_gap_inside_event" if overall_support else "no_supported_endpoint_clock_mapping")
    elif len(matches) > 1:
        flags.append("competing_clock_or_session_mappings")
    return {"status": "supported_observation_session" if not flags else "needs_source_review",
            "flags": ";".join(dict.fromkeys(flags)), "raw_session_count": len(sessions),
            "source_address_count": len(addresses), "mapping_names": ";".join(supported_names),
            "compatible_session_count": len(matches), "snap_span_minutes": span,
            "first_snap_epoch": first, "last_snap_epoch": last}, sessions


def read_month_sources(path, wanted):
    results = {}
    for chunk in pd.read_csv(path, usecols=["my_id", "time_recorded", "mode_s", "call_sign"],
                             dtype=str, keep_default_na=False, chunksize=250000):
        chunk["my_id"] = chunk.my_id.str.strip()
        chunk = chunk.loc[chunk.my_id.isin(wanted)].copy()
        if chunk.empty:
            continue
        clocks = pd.to_datetime(chunk.time_recorded, format="%d-%b-%Y %H:%M:%S", errors="coerce")
        chunk["valid_clock"] = clocks.notna()
        chunk["clock_seconds"] = clocks.astype("int64") // 10**9
        addresses = chunk.mode_s.str.strip().str.upper().str.replace(r"^0X", "", regex=True)
        chunk["valid_address"] = addresses.str.fullmatch(r"[0-9A-F]{6}")
        chunk["address"] = addresses
        for identifier, group in chunk.groupby("my_id", sort=False):
            item = results.setdefault(identifier, {"times_parts": [], "addresses": set(), "rows": 0,
                                                   "invalid_clocks": 0, "invalid_addresses": 0,
                                                   "populated_callsign_rows": 0})
            item["times_parts"].append(group.loc[group.valid_clock, "clock_seconds"].to_numpy(dtype=np.int64))
            item["addresses"].update(group.loc[group.valid_address, "address"])
            item["rows"] += len(group)
            item["invalid_clocks"] += int((~group.valid_clock).sum())
            item["invalid_addresses"] += int((~group.valid_address).sum())
            item["populated_callsign_rows"] += int(group.call_sign.str.strip().ne("").sum())
    for item in results.values():
        item["times"] = np.concatenate(item.pop("times_parts"))
    return results


def validate_sessions(project_dir, output_dir, prepared_file,
                      cleaned_dir=None, snapped_dir=None, programs_dir=None):
    project, output = Path(project_dir).resolve(), Path(output_dir).resolve()
    prepared_path = Path(prepared_file).resolve()
    cleaned = Path(cleaned_dir).resolve() if cleaned_dir else project / "Cleaned data"
    snapped_dir = Path(snapped_dir).resolve() if snapped_dir else project / "Snapped data"
    programs = Path(programs_dir).resolve() if programs_dir else project / "Programs"
    protected = [project / name for name in ["src", "tests", "Cleaned data", "Feature data", "Snapped data", "Programs"]]
    for path in protected:
        if output == path or output in path.parents or path in output.parents:
            raise ValueError("Use a separate Step 8 output directory; preserve source and previous experiments")
    inputs = {"features_prepared.csv": prepared_path}
    inputs.update({f"cleanedx_{i:02}.csv": cleaned / f"cleanedx_{i:02}.csv" for i in range(1, 13)})
    inputs.update({f"snapped{i}.csv": snapped_dir / f"snapped{i}.csv" for i in range(1, 13)})
    for name in ["cleaning_data.exe", "snap.exe", "feature_extract.exe", "man_map.txt"]:
        candidate = programs / name
        if candidate.is_file():
            inputs[name] = candidate
    configuration = {"policy": POLICY, "script_sha256": sha256(__file__),
                     "input_sha256": {name: sha256(path) for name, path in inputs.items()}}
    manifest_path = output / "session_validation_manifest.json"
    if output.exists() and any(output.iterdir()):
        if not manifest_path.is_file():
            raise ValueError("Partial Step 8 already exists; preserve it and use a new directory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("configuration") != configuration or set(manifest.get("output_sha256", {})) != OUTPUTS:
            raise ValueError("Step 8 provenance changed; use a new output directory")
        for name, digest in manifest["output_sha256"].items():
            if sha256(output / name) != digest:
                raise ValueError(f"Step 8 output changed: {name}")
        print("Existing source-session validation verified; outputs preserved.", flush=True)
        return json.loads((output / "session_validation_summary.json").read_text(encoding="utf-8"))
    prepared = pd.read_csv(prepared_path, usecols=["id", "type", "source_date", "source_month", "taxi_time"])
    if prepared.id.duplicated().any():
        raise ValueError("This validation requires one prepared feature record per daily ID")
    records, session_records, months = [], [], []
    for month in range(1, 13):
        part = prepared.loc[prepared.source_month.eq(month)]
        wanted = set(part.id)
        raw = read_month_sources(cleaned / f"cleanedx_{month:02}.csv", wanted)
        snap = pd.read_csv(snapped_dir / f"snapped{month}.csv", dtype={"flight_id": str})
        snap_groups = {identifier: group for identifier, group in snap.loc[snap.flight_id.isin(wanted)].groupby("flight_id", sort=False)}
        for row in part.itertuples(index=False):
            source_item = raw.get(row.id, {})
            status, sessions = classify_interval(row.taxi_time, source_item, snap_groups.get(row.id))
            records.append({"id": row.id, "type": row.type, "source_date": row.source_date,
                            "taxi_time": row.taxi_time, "source_month": month, "raw_rows": source_item.get("rows", 0),
                            "invalid_source_clock_rows": source_item.get("invalid_clocks", 0),
                            "invalid_or_missing_address_rows": source_item.get("invalid_addresses", 0),
                            "populated_callsign_rows": source_item.get("populated_callsign_rows", 0), **status})
            for index, (start, end, count) in enumerate(sessions):
                session_records.append({"id": row.id, "observation_session": index,
                                        "first_recorded_clock": pd.Timestamp(start, unit="s").isoformat(),
                                        "last_recorded_clock": pd.Timestamp(end, unit="s").isoformat(),
                                        "unique_clock_values": count, "clock_span_minutes": (end - start) / 60})
        months.append({"month": month, "prepared_records": len(part), "matched_source_rows": sum(item["rows"] for item in raw.values())})
        print(f"Source sessions: month {month:02}; {len(part):,} records checked", flush=True)
    table = pd.DataFrame(records).sort_values(["source_date", "id"], kind="stable")
    session_table = pd.DataFrame(session_records).sort_values(["id", "observation_session"], kind="stable")
    reason_counts = table.flags.str.split(";").explode() if False else table["flags"].str.split(";").explode()
    summary = {"policy": POLICY, "records_checked": len(table),
               "supported_observation_session_records": int(table.status.eq("supported_observation_session").sum()),
               "records_needing_source_review": int(table.status.eq("needs_source_review").sum()),
               "daily_ids_with_multiple_raw_sessions": int(table.raw_session_count.gt(1).sum()),
               "reason_counts": {str(k): int(v) for k, v in reason_counts[reason_counts.ne("")].value_counts().items()},
               "month_counts": months, "models_fitted": False, "existing_labels_changed": False,
               "notes": ["Source session support is not official taxi-event validation",
                         "Historical timezone is unresolved; candidate mappings are hypotheses",
                         "No duration cutoff, gap subtraction, error-based cleaning or fitting was performed",
                         "Original congestion features include original source intervals; any new accepted-event counts need separate reconstruction",
                         "This audit does not choose a revised model cohort or make the observed December dates fresh"]}
    for name, path in inputs.items():
        if sha256(path) != configuration["input_sha256"][name]:
            raise ValueError("Source input changed during validation; no outputs written")
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "record_validation.csv", index=False)
    session_table.to_csv(output / "source_sessions.csv", index=False)
    (output / "session_validation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps({"configuration": configuration, "output_sha256": {name: sha256(output / name) for name in sorted(OUTPUTS)}}, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepared-file", type=Path, required=True, help="Prepared table generated from the twelve monthly feature files")
    parser.add_argument("--cleaned-dir", type=Path, help="Cleaned source-observation directory")
    parser.add_argument("--snapped-dir", type=Path, help="Map-snapped source directory")
    parser.add_argument("--programs-dir", type=Path, help="Optional directory containing the supplied tools and airport map")
    args = parser.parse_args()
    try:
        summary = validate_sessions(args.project_dir, args.output_dir or args.project_dir / "artifacts/step08",
                                    args.prepared_file, args.cleaned_dir,
                                    args.snapped_dir, args.programs_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Source-session validation failed: {error}\n")
    print(f"Checked {summary['records_checked']:,} records; {summary['supported_observation_session_records']:,} supported under the observation-session policy; {summary['records_needing_source_review']:,} need source review.")
    print("No previous labels, features, models or scores were changed.")


if __name__ == "__main__":
    main()
