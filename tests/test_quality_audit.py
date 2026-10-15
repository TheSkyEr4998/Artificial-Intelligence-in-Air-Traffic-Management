import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quality_audit import development_baselines, interval_diagnostics, saved_error_diagnostics
from prediction_quality import input_quality, load_support_profile, training_support_profile


class QualityAuditTests(unittest.TestCase):
    def test_overlapping_intervals_are_unioned_and_uncovered_gap_retained(self):
        frame = pd.DataFrame({"index": [1, 1, 1], "t1": [0, 60, 1200], "t2": [120, 180, 1260]})
        result = interval_diagnostics(frame)
        self.assertEqual(result["span_minutes"], 21)
        self.assertEqual(result["union_interval_minutes"], 4)
        self.assertEqual(result["largest_uncovered_gap_minutes"], 17)
        self.assertEqual(result["longest_edge_minutes"], 2)

    def test_bad_timestamps_flagged_instead_of_silently_becoming_evidence(self):
        frame = pd.DataFrame({"index": [1, 2, 2], "t1": [2, np.nan, 10], "t2": [1, 3, np.inf]})
        result = interval_diagnostics(frame)
        self.assertEqual(result["invalid_interval_rows"], 3)
        self.assertEqual(result["snap_index_groups"], 2)
        self.assertIsNone(result["span_minutes"])

    def test_error_contribution_uses_saved_full_cohort_and_does_not_filter(self):
        frame = pd.DataFrame({"family": ["anfis", "anfis"], "actual_minutes": [1, 200],
                              "predicted_minutes": [2, 190]})
        result = saved_error_diagnostics(frame).set_index("duration_over_minutes")
        self.assertEqual(result.loc[120, "fraction_of_records"], .5)
        self.assertAlmostEqual(result.loc[120, "fraction_of_squared_error"], 100/101)
        self.assertAlmostEqual(result.loc[120, "fraction_of_absolute_error"], 10/11)

    def test_type_baseline_learns_medians_from_training_only(self):
        train = pd.DataFrame({"type": ["arrival", "arrival", "departure"], "taxi_time": [2, 4, 10]})
        validation = pd.DataFrame({"type": ["arrival", "departure"], "taxi_time": [100, 200]})
        result = development_baselines(train, validation)
        self.assertEqual(result["type_medians_minutes"], {"arrival": 3, "departure": 10})
        self.assertFalse(result["test_scored"])
        self.assertEqual(result["results"][1]["mae_minutes"], (97+190)/2)

    def test_missing_inputs_and_extrapolation_are_reported_separately(self):
        train = pd.DataFrame({"type": ["arrival", "departure"], "distance_gate": [1, 10], "start_hour": [0, 23]})
        factors = train.columns.tolist()
        profile = training_support_profile(train, factors, "digest")
        incoming = pd.DataFrame({"type": ["arrival", "departure"], "distance_gate": [np.nan, 11], "start_hour": [1, 3]})
        flags = input_quality(incoming, factors, profile)
        self.assertEqual(flags.imputed_factors.tolist(), ["distance_gate", ""])
        self.assertEqual(flags.outside_training_range_factors.tolist(), ["", "distance_gate"])
        self.assertEqual(flags.training_range_status.iloc[1], "outside_marginal_training_ranges")
        self.assertEqual(input_quality(incoming, factors).training_range_status.tolist(), ["not_checked"]*2)

    def test_support_profile_must_match_frozen_training_contract(self):
        train = pd.DataFrame({"type": ["arrival"], "distance_gate": [2]})
        profile = training_support_profile(train, list(train), "digest")
        selection = {"train_sha256": "digest", "selected_factors": list(train)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "support.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            self.assertEqual(load_support_profile(path, selection), profile)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_support_profile(path, {**selection, "train_sha256": "changed"})
            profile["bounds"]["distance_gate"]["minimum"] = 99
            path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid support"):
                load_support_profile(path, selection)


if __name__ == "__main__":
    unittest.main()
