import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from create_splits import FEATURES, NUMERIC_FEATURES, chronological_split, modelling_table, training_profile


def sample_data():
    rows = []
    # Growing traffic makes a split by date counts differ from a split by rows.
    for day_index, date in enumerate(pd.date_range("2021-01-01", periods=40)):
        for flight in range(1 + day_index // 4):
            row = {feature: 1.0 for feature in NUMERIC_FEATURES}
            row.update({
                "id": f"{date:%d-%b-%Y}_AIR{flight}", "source_date": date,
                "type": "arrival" if flight % 2 else "departure", "taxi_time": 10.0,
                "rwy": None, "review_reasons": "runway_name_missing",
            })
            rows.append(row)
    return pd.DataFrame(rows)


def memberships(data, assignment):
    return dict(zip(zip(data["id"], data["type"]), assignment))


class SplitTests(unittest.TestCase):
    def test_chronology_whole_date_grouping_and_complete_accounting(self):
        data, _, _ = modelling_table(sample_data())
        parts, assigned, details = chronological_split(data)
        combined = pd.concat(parts.values())
        self.assertEqual(len(combined), len(data))
        self.assertFalse(combined.duplicated(["id", "type"]).any())
        self.assertEqual(data.assign(split=assigned).groupby("source_date")["split"].nunique().max(), 1)
        self.assertLess(parts["train"].source_date.max(), parts["validation"].source_date.min())
        self.assertLess(parts["validation"].source_date.max(), parts["test"].source_date.min())
        for cutoff in [details["training_cutoff"], details["validation_cutoff"]]:
            gap_date = pd.Timestamp(cutoff) + pd.Timedelta(days=1)
            self.assertTrue(assigned.loc[data.source_date.eq(gap_date)].eq("boundary_gap").all())

    def test_boundaries_follow_row_counts_and_are_deterministic_under_shuffle(self):
        data, _, _ = modelling_table(sample_data())
        _, assigned, details = chronological_split(data, gap_days=0)
        shuffled = data.sample(frac=1, random_state=71).reset_index(drop=True)
        _, shuffled_assigned, shuffled_details = chronological_split(shuffled, gap_days=0)
        self.assertEqual(memberships(data, assigned), memberships(shuffled, shuffled_assigned))
        self.assertEqual(details, shuffled_details)
        counts = data.groupby("source_date").size()
        expected = (counts.cumsum().iloc[:-2] - 0.7 * len(data)).abs().idxmin()
        self.assertEqual(details["training_cutoff"], expected.strftime("%Y-%m-%d"))
        self.assertLess(abs(details["partitions"]["train"]["share_of_used_rows"] - .7), .04)

    def test_positive_target_changes_do_not_change_membership_and_extremes_remain(self):
        original = sample_data()
        changed = original.copy()
        changed.loc[0, "taxi_time"] = 0.1
        changed.loc[1, "taxi_time"] = 1400.0
        changed.loc[2, "shortest path"] = 0
        changed.loc[3, "angle"] = -1
        base, _, _ = modelling_table(original)
        data, provenance, _ = modelling_table(changed)
        _, a, _ = chronological_split(base)
        _, b, _ = chronological_split(data)
        self.assertEqual(memberships(base, a), memberships(data, b))
        self.assertEqual(len(data), len(original))
        self.assertIn(1400.0, data.taxi_time.tolist())
        self.assertIn(0.1, data.taxi_time.tolist())
        self.assertIn(0.0, data["shortest path"].tolist())
        self.assertIn(-1.0, data["angle"].tolist())
        self.assertTrue(provenance.review_reasons.str.contains("zero_shortest_path").any())
        self.assertTrue(provenance.review_reasons.str.contains("negative_angle").any())
        self.assertNotIn("rwy", data.columns)
        self.assertNotIn("review_reasons", data.columns)
        self.assertEqual(len(FEATURES), 10)

    def test_invalid_predictors_become_missing_with_original_values_audited(self):
        frame = sample_data()
        frame.loc[0, "distance"] = -100.0
        frame.loc[1, "start_hour"] = 24.0
        frame.loc[2, "n_departing_at_start"] = 1.5
        frame.loc[3, "angle"] = np.inf
        data, _, changes = modelling_table(frame)
        self.assertEqual(data[NUMERIC_FEATURES].isna().sum().sum(), 4)
        self.assertEqual(len(data), len(frame))
        self.assertEqual(set(changes.reason), {"negative_distance", "hour_not_integer_0_to_23",
                                               "count_not_nonnegative_integer", "nonfinite_numeric"})
        self.assertIn(-100.0, changes.original_value.tolist())

    def test_bad_identifiers_dates_targets_and_duplicates_fail_before_splitting(self):
        for column, invalid in [("id", None), ("source_date", "wrong date"),
                                ("taxi_time", 0), ("taxi_time", np.inf), ("type", "vehicle")]:
            with self.subTest(column=column, value=invalid):
                frame = sample_data()
                if column == "source_date":
                    frame[column] = frame[column].astype(object)
                frame.loc[0, column] = invalid
                with self.assertRaises(ValueError):
                    modelling_table(frame)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            modelling_table(pd.concat([sample_data(), sample_data().iloc[[0]]]))

    def test_calendar_gap_does_not_remove_the_next_observed_day_if_calendar_day_is_absent(self):
        data, _, _ = modelling_table(sample_data())
        # A sparse example keeps the original cutoffs predictable using widely spaced dates.
        data["source_date"] = pd.Timestamp("2021-01-01") + (data.source_date - pd.Timestamp("2021-01-01")) * 3
        parts, _, _ = chronological_split(data, gap_days=1)
        self.assertTrue(parts["boundary_gap"].empty)

    def test_too_few_dates_and_empty_partitions_after_gaps_fail_clearly(self):
        data, _, _ = modelling_table(sample_data())
        with self.assertRaisesRegex(ValueError, "three distinct"):
            chronological_split(data.loc[data.source_date.lt("2021-01-03")])
        with self.assertRaisesRegex(ValueError, "empty after date gaps"):
            chronological_split(data, gap_days=100)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            chronological_split(data, gap_days=-1)
        data.loc[0, "source_date"] = pd.NaT
        with self.assertRaisesRegex(ValueError, "dates must be valid"):
            chronological_split(data)

    def test_candidate_profile_uses_only_the_passed_training_records(self):
        data, _, _ = modelling_table(sample_data())
        parts, _, _ = chronological_split(data)
        expected = training_profile(parts["train"])
        parts["test"].loc[:, "distance"] = 99999
        pd.testing.assert_frame_equal(expected, training_profile(parts["train"]))
        self.assertEqual(expected.set_index("feature").loc["distance", "max"], 1.0)


if __name__ == "__main__":
    unittest.main()
