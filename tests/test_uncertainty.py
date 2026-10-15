"""Contracts for paired whole-day uncertainty of frozen test predictions."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uncertainty import cluster_bootstrap_metrics


class ClusterBootstrapTests(unittest.TestCase):
    def test_whole_days_are_kept_together_and_metrics_weight_records(self):
        dates = ["2021-01-01", *["2021-01-02"] * 3]
        actual = np.array([10., 11., 12., 13.])
        predictions = {
            "median_baseline": actual - [0, 8, 8, 8],
            "polynomial": actual - [2, 4, 4, 4],
        }
        result = cluster_bootstrap_metrics(dates, actual, predictions, "polynomial")
        baseline = result["models"]["median_baseline"]
        winner = result["models"]["polynomial"]
        self.assertEqual(baseline["mae_minutes"]["point"], 6.)
        self.assertAlmostEqual(baseline["rmse_minutes"]["point"], np.sqrt(48))
        self.assertAlmostEqual(baseline["r2"]["point"], 1 - 192 / 5)
        self.assertEqual(winner["mae_minutes"]["point"], 3.5)
        # The quiet date is drawn twice with probability 1/4, retaining both
        # zero errors. An IID-row bootstrap would put its 2.5th percentile above
        # zero because drawing this one record four times has probability 1/256.
        self.assertEqual(baseline["mae_minutes"]["lower95"], 0.)
        self.assertEqual(baseline["mae_minutes"]["upper95"], 8.)
        self.assertEqual(baseline["rmse_minutes"]["lower95"], 0.)
        self.assertEqual(baseline["rmse_minutes"]["upper95"], 8.)
        difference = result["paired_mae_differences"]["median_baseline"]
        self.assertEqual(difference["point"], -2.5)
        self.assertEqual(difference["lower95"], -4.)
        self.assertEqual(difference["upper95"], 2.)
        self.assertEqual(result["n_days"], 2)
        self.assertEqual(result["n_rows"], 4)

    def test_identical_predictions_have_exactly_zero_paired_differences(self):
        actual = np.array([1., 2., 3., 4.])
        predicted = np.array([2., 1., 3., 5.])
        result = cluster_bootstrap_metrics(
            ["2021-01-01"] * 2 + ["2021-01-02"] * 2, actual,
            {"anfis": predicted.copy(), "polynomial": predicted.copy(), "median_baseline": predicted.copy()},
            "polynomial", iterations=600,
        )
        for interval in result["paired_mae_differences"].values():
            self.assertEqual([interval[name] for name in ["point", "lower95", "upper95"]], [0., 0., 0.])
            self.assertEqual(interval["valid_replicates"], 600)
            self.assertEqual(interval["invalid_replicates"], 0)

    def test_r2_uses_each_replicates_target_mean_and_counts_undefined_draws(self):
        dates = ["2021-01-01"] * 2 + ["2021-01-02"] * 2
        actual = np.array([0., 0., 2., 2.])
        result = cluster_bootstrap_metrics(dates, actual, {"polynomial": actual + 1}, "polynomial", iterations=600)
        r2 = result["models"]["polynomial"]["r2"]
        # Mixed-day draws have R2 = 0. Single-day draws have constant targets,
        # so R2 is undefined rather than being evaluated about the global mean.
        self.assertEqual([r2[key] for key in ["point", "lower95", "upper95"]], [0., 0., 0.])
        self.assertGreater(r2["invalid_replicates"], 0)
        self.assertGreater(r2["valid_replicates"], 0)
        self.assertEqual(r2["valid_replicates"] + r2["invalid_replicates"], 600)
        self.assertEqual(r2["interval_population"], "defined_replicates_only")

    def test_constant_targets_have_no_forced_finite_r2_or_invalid_json(self):
        dates = ["2021-01-01"] * 3 + ["2021-01-02"] * 3
        for value in [5., .1]:
            with self.subTest(constant=value):
                # The decimal case also catches a tiny floating-point difference
                # between a constant value and its computed arithmetic mean.
                actual = np.full(6, value)
                result = cluster_bootstrap_metrics(dates, actual, {"polynomial": actual + 1}, "polynomial", iterations=100)
                r2 = result["models"]["polynomial"]["r2"]
                self.assertIsNone(r2["point"])
                self.assertIsNone(r2["lower95"])
                self.assertIsNone(r2["upper95"])
                self.assertEqual(r2["invalid_replicates"], 100)
                self.assertEqual(r2["valid_replicates"], 0)
                json.dumps(result, allow_nan=False)

    def test_seed_and_shared_row_permutation_are_reproducible_without_mutation(self):
        dates = pd.Series(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-03"], index=[8, 2, 7, 3])
        actual = np.array([1., 2., 3., 4.])
        predictions = {"polynomial": actual + [1, 1, 2, 3], "median_baseline": actual + 2}
        dates_before = dates.copy(deep=True)
        target_before = actual.copy()
        predictions_before = {name: values.copy() for name, values in predictions.items()}
        first = cluster_bootstrap_metrics(dates, actual, predictions, "polynomial", iterations=300, seed=17)
        repeat = cluster_bootstrap_metrics(dates, actual, predictions, "polynomial", iterations=300, seed=17)
        permutation = np.array([3, 1, 0, 2])
        permuted = cluster_bootstrap_metrics(dates.iloc[permutation], actual[permutation],
                                             {name: values[permutation] for name, values in reversed(list(predictions.items()))},
                                             "polynomial", iterations=300, seed=17)
        self.assertEqual(first, repeat)
        self.assertEqual(first, permuted)
        pd.testing.assert_series_equal(dates, dates_before)
        np.testing.assert_array_equal(actual, target_before)
        for name in predictions:
            np.testing.assert_array_equal(predictions[name], predictions_before[name])

    def test_one_date_does_not_produce_false_between_day_precision(self):
        with self.assertRaisesRegex(ValueError, "two distinct source dates"):
            cluster_bootstrap_metrics(["2021-01-01"] * 3, [1, 2, 3], {"polynomial": [1, 2, 4]}, "polynomial")

    def test_invalid_keys_values_lengths_dates_or_random_settings_are_rejected(self):
        dates, actual = ["2021-01-01", "2021-01-02"], np.array([1., 2.])
        arguments = [
            (dates, actual, {"polynomial": [1., np.inf]}, "polynomial", {}, "finite vector"),
            (dates, [1., np.nan], {"polynomial": [1., 2.]}, "polynomial", {}, "finite one-dimensional"),
            (dates, actual, {"polynomial": [1.]}, "polynomial", {}, "target row count"),
            (dates, actual, {"polynomial": [[1.], [2.]]}, "polynomial", {}, "finite vector"),
            (dates, actual, {7: [1., 2.]}, "polynomial", {}, "family keys"),
            (dates, actual, {"": [1., 2.]}, "polynomial", {}, "family keys"),
            (dates, actual, {}, "polynomial", {}, "nonempty mapping"),
            (dates, actual, {"polynomial": [1., 2.]}, "unknown", {}, "winner"),
            (["2021-01-01", None], actual, {"polynomial": [1., 2.]}, "polynomial", {}, "calendar dates"),
            (["bad-date", "2021-01-02"], actual, {"polynomial": [1., 2.]}, "polynomial", {}, "calendar dates"),
            ([0, 86_400_000_000_000], actual, {"polynomial": [1., 2.]}, "polynomial", {}, "numeric timestamp keys"),
            (dates[:1], actual, {"polynomial": [1., 2.]}, "polynomial", {}, "same row count"),
            (dates, actual, {"polynomial": [1., 2.]}, "polynomial", {"iterations": True}, "positive integer"),
            (dates, actual, {"polynomial": [1., 2.]}, "polynomial", {"seed": -1}, "nonnegative integer"),
        ]
        for date_values, y, predictions, winner, options, message in arguments:
            with self.subTest(message=message, options=options):
                with self.assertRaisesRegex(ValueError, message):
                    cluster_bootstrap_metrics(date_values, y, predictions, winner, **options)


if __name__ == "__main__":
    unittest.main()
