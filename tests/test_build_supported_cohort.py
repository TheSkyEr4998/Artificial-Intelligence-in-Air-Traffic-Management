import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.build_supported_cohort import build_supported_cohort


class SupportedCohortTests(unittest.TestCase):
    def test_only_source_supported_rows_are_kept_and_labels_are_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared.csv"
            validation = root / "validation.csv"
            output = root / "cohort"
            pd.DataFrame([
                {"id": "01-Jan-2021_A", "type": "arrival", "source_date": "2021-01-01",
                 "taxi_time": 8.5, "source_month": 1, "distance": 100.0},
                {"id": "02-Jan-2021_B", "type": "departure", "source_date": "2021-01-02",
                 "taxi_time": 12.0, "source_month": 1, "distance": 200.0},
            ]).to_csv(prepared, index=False)
            pd.DataFrame([
                {"id": "01-Jan-2021_A", "type": "arrival", "status": "supported_observation_session",
                 "flags": "", "mapping_names": "UTC_clock", "raw_session_count": 1,
                 "source_address_count": 1, "compatible_session_count": 1, "snap_span_minutes": 8.5},
                {"id": "02-Jan-2021_B", "type": "departure", "status": "needs_source_review",
                 "flags": "observation_gap_inside_event", "mapping_names": "", "raw_session_count": 2,
                 "source_address_count": 1, "compatible_session_count": 0, "snap_span_minutes": 12.0},
            ]).to_csv(validation, index=False)
            summary = build_supported_cohort(prepared, validation, output)
            kept = pd.read_csv(output / "features_supported.csv")
            dropped = pd.read_csv(output / "features_for_source_review.csv")
            contract = json.loads((output / "cohort_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["supported_rows"], 1)
            self.assertEqual(summary["excluded_for_source_review_rows"], 1)
            self.assertEqual(kept.iloc[0].id, "01-Jan-2021_A")
            self.assertEqual(kept.iloc[0].taxi_time, 8.5)
            self.assertEqual(dropped.iloc[0].taxi_time, 12.0)
            self.assertEqual(contract["target_or_predictors_modified"], False)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared.csv"
            validation = root / "validation.csv"
            output = root / "cohort"
            pd.DataFrame([{"id": "01-Jan-2021_A", "type": "arrival", "source_date": "2021-01-01"}]).to_csv(prepared, index=False)
            pd.DataFrame([{"id": "01-Jan-2021_A", "type": "arrival", "status": "supported_observation_session",
                           "flags": "", "mapping_names": "UTC_clock", "raw_session_count": 1,
                           "source_address_count": 1, "compatible_session_count": 1,
                           "snap_span_minutes": 1.0}]).to_csv(validation, index=False)
            output.mkdir()
            sentinel = output / "preserve.txt"
            sentinel.write_text("keep")
            with self.assertRaises(ValueError):
                build_supported_cohort(prepared, validation, output)
            self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
