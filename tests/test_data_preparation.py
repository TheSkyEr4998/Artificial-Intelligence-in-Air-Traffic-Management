import sys
import unittest
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_preparation import RAW_COLUMNS, prepare_records, save_preparation


def record(identifier="02-Jan-2021_AAL123", movement="departure", month=1, duration=10):
    row = {column: 0 for column in RAW_COLUMNS}
    row.update({
        "id": identifier, "type": movement, "taxi_time": duration, "rwy": "23R/05L",
        "gate (block) hour": 8, "runway hour": 9,
        "NDepDep": 2, "NDepArr": 3, "NArrDep": 4, "NArrArr": 5,
        "source_file": f"month-{month}.csv", "source_month": month,
    })
    return row


class PreparationTests(unittest.TestCase):
    def test_identity_quarantine_is_independent_of_target_duration(self):
        conflict = "02-Jan-2021_AAL123"
        rows = [record(conflict, duration=3), record("02-Jan-2021_AAL124", duration=300)]
        prepared, excluded, _, summary = prepare_records(pd.DataFrame(rows), ambiguous_identifiers={conflict})
        self.assertEqual(prepared.id.tolist(), ["02-Jan-2021_AAL124"])
        self.assertEqual(excluded.exclusion_reason.tolist(), ["ambiguous_aircraft_identity_for_daily_id"])
        self.assertEqual(summary["preparation_policy_version"], 3)
    def test_icelandair_flights_retained_but_spaced_ice_vehicles_excluded(self):
        aircraft = ["ICE440", "ICE441", "ICE1041", "ICE1042"]
        rows = [record(f"02-Jan-2021_{identifier}") for identifier in aircraft + ["ICE 1", "ICE 2"]]
        prepared, excluded, _, _ = prepare_records(pd.DataFrame(rows))
        self.assertEqual(set(prepared.source_identifier), set(aircraft))
        self.assertEqual(set(excluded.source_identifier), {"ICE 1", "ICE 2"})

    def test_changed_preparation_cannot_overwrite_previous_benchmark(self):
        merged = pd.DataFrame([record()])
        prepared, excluded, review, summary = prepare_records(merged)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_preparation(root, merged, prepared, excluded, review, summary)
            original = (root / "features_prepared.csv").read_bytes()
            save_preparation(root, merged, prepared, excluded, review, summary)
            with self.assertRaisesRegex(ValueError, "Existing preparation differs"):
                save_preparation(root, merged, prepared.assign(taxi_time=99), excluded, review, summary)
            self.assertEqual((root / "features_prepared.csv").read_bytes(), original)

    def test_calendar_owner_resolves_overlap_without_arbitrary_conflicting_choice(self):
        previous_month = record("01-Feb-2021_AAL123", month=1, duration=300)
        current_month = record("01-Feb-2021_AAL123", month=2, duration=10)
        prepared, excluded, _, summary = prepare_records(pd.DataFrame([previous_month, current_month]))
        self.assertEqual(prepared["taxi_time"].tolist(), [10])
        self.assertEqual(excluded["exclusion_reason"].tolist(), ["month_boundary_overlap"])
        self.assertEqual(summary["prepared_rows"], 1)

    def test_invalid_labels_and_named_vehicles_removed_without_excluding_unknown_aircraft_ids(self):
        rows = [record(duration=-2), record("02-Jan-2021_OPS 3"), record("02-Jan-2021_484B91")]
        prepared, excluded, _, _ = prepare_records(pd.DataFrame(rows))
        self.assertEqual(prepared["id"].tolist(), ["02-Jan-2021_484B91"])
        self.assertEqual(set(excluded["exclusion_reason"]),
                         {"nonpositive_or_missing_taxi_time", "known_vehicle_identifier"})

    def test_direction_normalization_uses_start_side_only(self):
        rows = [record(), record("02-Jan-2021_AAL124", movement="arrival")]
        prepared, _, _, _ = prepare_records(pd.DataFrame(rows))
        departure = prepared.loc[prepared["type"].eq("departure")].iloc[0]
        arrival = prepared.loc[prepared["type"].eq("arrival")].iloc[0]
        self.assertEqual((departure.start_hour, departure.n_departing_at_start, departure.n_arriving_at_start), (8, 2, 3))
        self.assertEqual((arrival.start_hour, arrival.n_departing_at_start, arrival.n_arriving_at_start), (9, 4, 5))

    def test_long_duration_is_flagged_and_preserved_for_review(self):
        prepared, excluded, review, _ = prepare_records(pd.DataFrame([record(duration=200)]))
        self.assertEqual(len(prepared), 1)
        self.assertTrue(excluded.empty)
        self.assertIn("duration_over_120_minutes", review.review_reasons.iloc[0])

    def test_conflicting_records_in_the_canonical_month_require_investigation(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            prepare_records(pd.DataFrame([record(duration=10), record(duration=20)]))


if __name__ == "__main__":
    unittest.main()
