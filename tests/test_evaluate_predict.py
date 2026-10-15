import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIR))
from anfis import ANFISRegressor
from create_splits import TARGET, sha256_file
from evaluate import evaluate_models, heldout_metrics
from model_io import load_frozen_models, predict_from_artifact, validate_prediction_inputs
from predict import normalize_start_factors, predict_file
from prediction_quality import training_support_profile
from train_models import train_and_select
from test_train_models import development_fixture, synthetic_records


def write_plot_placeholders(test, predictions, winner, metrics, intervals, output_dir):
    """Keep manifest checks real while excluding plot rendering from these tests."""
    for name in ["test_model_comparison.png", "winner_test_scatter.png",
                 "winner_test_residuals.png", "test_absolute_error_cdf.png"]:
        (output_dir / name).write_bytes(b"test-only plot placeholder")


@contextlib.contextmanager
def forbid_fitting():
    with contextlib.ExitStack() as stack:
        for estimator in [TransformedTargetRegressor, DummyRegressor, StandardScaler,
                          Ridge, MLPRegressor, KMeans, ANFISRegressor]:
            stack.enter_context(patch.object(estimator, "fit", side_effect=AssertionError("Evaluation/inference must not fit")))
        stack.enter_context(patch("train_models.load_training_inputs", side_effect=AssertionError("No training during evaluation")))
        yield


class EvaluatePredictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared_temporary = tempfile.TemporaryDirectory()
        cls.fixture_root = Path(cls.shared_temporary.name) / "fixture"
        cls.fixture_root.mkdir()
        split_dir, features_dir, train, validation = development_fixture(cls.fixture_root)
        test = pd.concat([
            synthetic_records(8, f"2021-03-0{day}", f"test{day}", 40 + day)
            for day in [1, 2, 3]
        ], ignore_index=True)
        test.to_csv(split_dir / "test.csv", index=False)
        prepared_dir = cls.fixture_root / "step01"
        prepared_dir.mkdir()
        prepared_path = prepared_dir / "features_prepared.csv"
        pd.concat([train, validation, test], ignore_index=True).to_csv(prepared_path, index=False)
        manifest_path = split_dir / "split_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"validation_cutoff": "2021-02-01", "gap_days": 1,
                         "input_sha256": sha256_file(prepared_path)})
        manifest["partitions"]["test"] = {
            "rows": len(test), "date_groups": 3,
            "first_date": "2021-03-01", "last_date": "2021-03-03",
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        contract_path = features_dir / "selected_features.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["split_manifest_sha256"] = sha256_file(manifest_path)
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        with patch("train_models.make_training_plots"), \
             patch.dict(os.environ, {"LOKY_MAX_CPU_COUNT": "2"}), \
             contextlib.redirect_stdout(io.StringIO()):
            cls.selection = train_and_select(split_dir, features_dir, cls.fixture_root / "step04", quick=True)

    @classmethod
    def tearDownClass(cls):
        cls.shared_temporary.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "copy"
        shutil.copytree(self.fixture_root, self.root)
        self.split_dir = self.root / "step02"
        self.features_dir = self.root / "step03"
        self.model_dir = self.root / "step04"
        self.output_dir = self.root / "step05"
        self.prepared_path = self.root / "step01" / "features_prepared.csv"

    def tearDown(self):
        self.temporary.cleanup()

    def evaluate(self, **kwargs):
        with patch("evaluate.make_test_plots", side_effect=write_plot_placeholders), \
             contextlib.redirect_stdout(io.StringIO()):
            return evaluate_models(self.split_dir, self.features_dir, self.model_dir,
                                   self.output_dir, iterations=kwargs.pop("iterations", 100), **kwargs)

    def test_final_evaluation_scores_actual_frozen_models_without_any_fit(self):
        model_digests = self.selection["fitted_model_sha256"].copy()
        with forbid_fitting():
            result = self.evaluate()
        self.assertFalse(result["test_used_for_model_selection"])
        self.assertFalse(result["refit_during_evaluation"])
        self.assertEqual(result["winner_selected_on_validation"], self.selection["winner_family"])
        self.assertEqual(result["test_rows"], 24)
        self.assertEqual(result["test_days"], 3)
        self.assertEqual(set(result["metrics"]), {"polynomial", "neural_network", "anfis", "median_baseline"})
        for family, digest in model_digests.items():
            self.assertEqual(sha256_file(self.model_dir / f"{family}.joblib"), digest)
        manifest = json.loads((self.output_dir / "evaluation_manifest.json").read_text())
        self.assertEqual(manifest["configuration"]["winner_family"], self.selection["winner_family"])
        self.assertEqual(len(manifest["output_sha256"]), 11)
        predictions = pd.read_csv(self.output_dir / "test_predictions.csv")
        self.assertEqual(len(predictions), 24 * 4)
        self.assertTrue(np.isfinite(predictions.predicted_minutes).all())

    def test_descriptive_test_ranking_never_replaces_the_validation_winner(self):
        winner = self.selection["winner_family"]
        alternative = next(family for family in self.selection["fitted_model_sha256"] if family != winner)

        def test_ranking_reversal(frame, artifact):
            return frame[TARGET].to_numpy() + (50 if artifact["family"] == winner else 0)

        with patch("evaluate.predict_from_artifact", side_effect=test_ranking_reversal):
            result = self.evaluate()
        self.assertEqual(result["winner_selected_on_validation"], winner)
        self.assertEqual(result["metrics"][alternative]["mae_minutes"], 0)
        self.assertEqual(result["metrics"][winner]["mae_minutes"], 50)

    def test_test_labels_or_predictors_changed_after_training_are_rejected(self):
        original = pd.read_csv(self.split_dir / "test.csv")
        for column in [TARGET, "distance"]:
            with self.subTest(column=column):
                changed = original.copy()
                changed.loc[0, column] += 1
                changed.to_csv(self.split_dir / "test.csv", index=False)
                with self.assertRaisesRegex(ValueError, "no longer match the prepared source"), \
                     patch("evaluate.predict_from_artifact", side_effect=AssertionError("No scoring changed test data")):
                    self.evaluate()
        original.to_csv(self.split_dir / "test.csv", index=False)
        with self.prepared_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "Prepared source changed"):
            self.evaluate()

    def test_model_corruption_is_rejected_before_unpickling(self):
        model_path = self.model_dir / "polynomial.joblib"
        with model_path.open("ab") as handle:
            handle.write(b"corrupted")
        with patch("model_io.joblib.load", side_effect=AssertionError("No changed artifact may be unpickled")):
            with self.assertRaisesRegex(ValueError, "Frozen model changed"):
                load_frozen_models(self.model_dir)

    def test_identical_evaluation_reuses_cache_and_rejects_configuration_or_output_edits(self):
        first = self.evaluate()
        with patch("evaluate.read_verified_test", side_effect=AssertionError("Cached evaluation must not reread labels")), \
             patch("evaluate.predict_from_artifact", side_effect=AssertionError("Cached evaluation must not predict again")):
            self.assertEqual(self.evaluate(), first)
        with self.assertRaisesRegex(ValueError, "different build/input/configuration"):
            self.evaluate(iterations=101)
        with (self.output_dir / "test_metrics.csv").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "Completed evaluation output changed"):
            self.evaluate()

    def test_prediction_validation_rejects_bad_values_but_imputes_missing_cells(self):
        _, artifacts = load_frozen_models(self.model_dir)
        artifact = artifacts[self.selection["winner_family"]]
        frame = pd.read_csv(self.split_dir / "validation.csv").iloc[:3].copy()
        factors = artifact["selected_factors"]
        for column, value in [("distance", -1), ("shortest path", -1),
                              ("n_departing_at_start", -.5), ("n_arriving_at_start", 1.5),
                              ("start_hour", 24), ("start_hour", 3.2),
                              ("distance", np.inf), ("type", "vehicle")]:
            with self.subTest(column=column, value=value):
                changed = frame.copy()
                if column != "type":
                    changed[column] = changed[column].astype(float)
                changed.loc[changed.index[0], column] = value
                with self.assertRaises(ValueError):
                    validate_prediction_inputs(changed, factors)
        with self.assertRaisesRegex(ValueError, "columns missing"):
            validate_prediction_inputs(frame.drop(columns=["distance"]), factors)
        with self.assertRaisesRegex(ValueError, "no records"):
            validate_prediction_inputs(frame.iloc[:0], factors)
        missing = frame.copy()
        missing.loc[missing.index[0], ["distance", "start_hour", "n_departing_at_start"]] = np.nan
        missing.loc[missing.index[1], "type"] = " DEPARTURE "
        self.assertTrue(np.isfinite(predict_from_artifact(missing, artifact)).all())
        changed_metadata = frame.assign(taxi_time=-1e99, id="ignored", source_date="invalid metadata")
        np.testing.assert_array_equal(predict_from_artifact(frame, artifact),
                                      predict_from_artifact(changed_metadata, artifact))

    def test_raw_schema_normalization_uses_start_side_for_each_direction(self):
        raw = pd.DataFrame({
            "type": [" Departure ", " ARRIVAL "],
            "gate (block) hour": [2, 3], "runway hour": [20, 21],
            "NDepDep": [4, 5], "NArrDep": [14, 15],
            "NDepArr": [6, 7], "NArrArr": [16, 17],
        })
        normalized = normalize_start_factors(raw)
        self.assertEqual(normalized.start_hour.tolist(), [2, 21])
        self.assertEqual(normalized.n_departing_at_start.tolist(), [4, 15])
        self.assertEqual(normalized.n_arriving_at_start.tolist(), [6, 17])
        changed = raw.copy()
        changed.loc[0, ["runway hour", "NArrDep", "NArrArr"]] = 999
        changed.loc[1, ["gate (block) hour", "NDepDep", "NDepArr"]] = 999
        again = normalize_start_factors(changed)
        columns = ["start_hour", "n_departing_at_start", "n_arriving_at_start"]
        pd.testing.assert_frame_equal(normalized[columns], again[columns])
        explicit = raw.assign(start_hour=[8, 9])
        self.assertEqual(normalize_start_factors(explicit).start_hour.tolist(), [8, 9])

    def test_prediction_file_and_fresh_process_cli_match_without_fitting(self):
        input_path = self.root / "new_feature_records.csv"
        frame = pd.read_csv(self.split_dir / "validation.csv").iloc[:5]
        frame.drop(columns=[TARGET, "id", "source_date"]).to_csv(input_path, index=False)
        output_path = self.root / "predictions.csv"
        with forbid_fitting(), contextlib.redirect_stdout(io.StringIO()):
            predicted = predict_file(input_path, output_path, self.model_dir)
        metadata = json.loads((self.root / "predictions_metadata.json").read_text())
        self.assertFalse(metadata["target_used_as_predictor"])
        self.assertEqual(metadata["rows"], 5)
        self.assertEqual(metadata["family"], self.selection["winner_family"])
        self.assertEqual(metadata["output_sha256"], sha256_file(output_path))
        cli_output = self.root / "cli_predictions.csv"
        completed = subprocess.run([
            sys.executable, str(SOURCE_DIR / "predict.py"), "--input", str(input_path),
            "--output", str(cli_output), "--model-dir", str(self.model_dir),
        ], capture_output=True, text=True, timeout=45)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        observed = pd.read_csv(cli_output)
        np.testing.assert_allclose(observed.predicted_taxi_time_minutes,
                                   predicted.predicted_taxi_time_minutes, atol=1e-10)
        self.assertEqual(observed.model_family.tolist(), predicted.model_family.tolist())
        with self.assertRaisesRegex(ValueError, "must differ"):
            predict_file(input_path, input_path, self.model_dir)

    def test_prediction_file_marks_imputation_and_range_without_changing_predictions(self):
        train = pd.read_csv(self.split_dir / "train.csv")
        factors = self.selection["selected_factors"]
        profile = training_support_profile(train, factors, self.selection["train_sha256"])
        profile_path = self.root / "support.json"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        frame = pd.read_csv(self.split_dir / "validation.csv").iloc[:2].copy()
        frame.loc[frame.index[0], "distance"] = np.nan
        frame.loc[frame.index[1], "distance"] = profile["bounds"]["distance"]["maximum"] + 100
        input_path = self.root / "incoming.csv"
        frame.to_csv(input_path, index=False)
        with forbid_fitting(), contextlib.redirect_stdout(io.StringIO()):
            output = predict_file(input_path, self.root / "flagged.csv", self.model_dir, support_profile=profile_path)
            plain = predict_file(input_path, self.root / "plain.csv", self.model_dir)
        np.testing.assert_array_equal(output.predicted_taxi_time_minutes, plain.predicted_taxi_time_minutes)
        self.assertIn("distance", output.imputed_factors.iloc[0])
        self.assertIn("distance", output.outside_training_range_factors.iloc[1])
        self.assertEqual(plain.training_range_status.tolist(), ["not_checked", "not_checked"])

    def test_heldout_metrics_keep_negative_predictions_and_expose_undefined_r2(self):
        score = heldout_metrics([2, 2], [-1, 2])
        self.assertEqual(score["mae_minutes"], 1.5)
        self.assertEqual(score["negative_prediction_count"], 1)
        self.assertIsNone(score["r2"])
        for actual, prediction in [([0, 1], [1, 1]), ([1, 2], [1, np.inf]), ([1, 2], [1])]:
            with self.assertRaisesRegex(ValueError, "Invalid held-out"):
                heldout_metrics(actual, prediction)


if __name__ == "__main__":
    unittest.main()
