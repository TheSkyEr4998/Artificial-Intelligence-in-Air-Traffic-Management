import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIR))
from create_splits import FEATURES, TARGET, sha256_file
from preprocessing import CONTEXT, fit_encoder, transform_features
from train_models import (
    candidate_grid, choice_key, load_training_inputs, predict_minutes,
    target_model, train_and_select,
)


def synthetic_records(count, date, prefix, seed):
    rng = np.random.default_rng(seed)
    gate = rng.uniform(10, 150, count)
    long = rng.uniform(200, 1400, count)
    other = rng.uniform(20, 600, count)
    distance = gate + long + other
    congestion = rng.integers(0, 12, count)
    return pd.DataFrame({
        "id": [f"{prefix}-{i}" for i in range(count)], "source_date": date,
        "type": ["arrival" if i % 2 == 0 else "departure" for i in range(count)],
        TARGET: 2 + distance / 160 + congestion / 3 + rng.uniform(0, 1, count),
        "start_hour": np.arange(count) % 24,
        "n_departing_at_start": congestion,
        "n_arriving_at_start": rng.integers(0, 12, count),
        "distance": distance, "angle": rng.uniform(0, 360, count),
        "shortest path": rng.uniform(200, 1800, count),
        "distance_gate": gate, "distance_long": long, "distance_else": other,
    })


def development_fixture(directory):
    split_dir, features_dir = directory / "step02", directory / "step03"
    split_dir.mkdir()
    features_dir.mkdir()
    train = synthetic_records(48, "2021-01-01", "train", 29)
    validation = synthetic_records(16, "2021-02-01", "validation", 31)
    manifest = {
        "prediction_task": "retrospective_completed_journey_benchmark",
        "candidate_features": FEATURES,
        "partitions": {
            "train": {"rows": len(train), "first_date": "2021-01-01", "last_date": "2021-01-01"},
            "validation": {"rows": len(validation), "first_date": "2021-02-01", "last_date": "2021-02-01"},
        },
    }
    for name, frame in [("train", train), ("validation", validation)]:
        frame.to_csv(split_dir / f"{name}.csv", index=False)
    (split_dir / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # Fit to the saved/reloaded values, just as Step 3 does, to make the frozen
    # provenance independent of CSV float-roundtrip precision.
    factors = [*CONTEXT, "distance", "angle", "shortest path"]
    encoder = fit_encoder(pd.read_csv(split_dir / "train.csv"), factors)
    contract = {
        "prediction_task": manifest["prediction_task"],
        "selected_factors": factors, "encoded_columns": encoder["encoded_columns"],
        "train_sha256": sha256_file(split_dir / "train.csv"),
        "validation_sha256": sha256_file(split_dir / "validation.csv"),
        "split_manifest_sha256": sha256_file(split_dir / "split_manifest.json"),
    }
    (features_dir / "encoder.json").write_text(json.dumps(encoder), encoding="utf-8")
    (features_dir / "selected_features.json").write_text(json.dumps(contract), encoding="utf-8")
    return split_dir, features_dir, train, validation


class TrainModelTests(unittest.TestCase):
    def test_loader_uses_only_train_and_validation_with_no_test_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            split_dir, features_dir, _, _ = development_fixture(Path(temporary))
            with patch("preprocessing.pd.read_csv", wraps=pd.read_csv) as reader:
                train, validation, x_train, x_validation, encoder, contract, _ = load_training_inputs(split_dir, features_dir)
            self.assertEqual([Path(call.args[0]).name for call in reader.call_args_list],
                             ["train.csv", "validation.csv"])
            self.assertEqual(x_train.shape, (len(train), 8))
            self.assertEqual(x_validation.shape, (len(validation), 8))
            self.assertEqual(encoder["factors"], contract["selected_factors"])
            self.assertFalse((split_dir / "test.csv").exists())

    def test_loader_rejects_changed_training_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            split_dir, features_dir, _, _ = development_fixture(Path(temporary))
            with (split_dir / "train.csv").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "train.csv changed since Step 3"):
                load_training_inputs(split_dir, features_dir)

    def test_loader_rejects_edited_encoder_imputation_or_scaling(self):
        with tempfile.TemporaryDirectory() as temporary:
            split_dir, features_dir, _, _ = development_fixture(Path(temporary))
            encoder_path = features_dir / "encoder.json"
            original = json.loads(encoder_path.read_text(encoding="utf-8"))
            changed = json.loads(json.dumps(original))
            changed["numeric_medians"]["distance"] += 100
            encoder_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "imputation values"):
                load_training_inputs(split_dir, features_dir)
            changed = json.loads(json.dumps(original))
            changed["scale"][0] += 1
            encoder_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scaling values"):
                load_training_inputs(split_dir, features_dir)

    def test_target_transform_is_log_then_training_scaling_and_inverse_returns_minutes(self):
        x_train = np.arange(12, dtype=float).reshape(6, 2)
        y_train = np.array([1, 2, 4, 8, 16, 32], dtype=float)
        model = target_model(DummyRegressor(strategy="mean")).fit(x_train, y_train)
        scaler = model.transformer_["scale"]
        logs = np.log1p(y_train)
        np.testing.assert_allclose(scaler.mean_, [logs.mean()])
        np.testing.assert_allclose(scaler.scale_, [logs.std()])
        self.assertEqual(int(scaler.n_samples_seen_), len(y_train))
        np.testing.assert_allclose(model.regressor_.constant_, [[0]], atol=1e-12)
        predicted = predict_minutes(model, np.full((3, 2), 1e9))
        np.testing.assert_allclose(predicted, np.expm1(logs.mean()))
        self.assertGreater(abs(predicted[0] - y_train.mean()), 1)
        before = (scaler.mean_.copy(), scaler.scale_.copy())
        predict_minutes(model, np.full((3, 2), -1e9))
        np.testing.assert_array_equal(scaler.mean_, before[0])
        np.testing.assert_array_equal(scaler.scale_, before[1])

    def test_changed_validation_values_do_not_change_training_encoding_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            split_dir, features_dir, _, _ = development_fixture(Path(temporary))
            before = load_training_inputs(split_dir, features_dir)
            validation_path = split_dir / "validation.csv"
            changed = pd.read_csv(validation_path)
            changed["distance"] = 1e8
            changed[TARGET] = 1e5
            changed.to_csv(validation_path, index=False)
            # Create a matching development contract for the perturbed fixture;
            # the separate fingerprint test checks that silent edits are rejected.
            contract_path = features_dir / "selected_features.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["validation_sha256"] = sha256_file(validation_path)
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            after = load_training_inputs(split_dir, features_dir)
            self.assertEqual(before[4], after[4])
            np.testing.assert_array_equal(before[2], after[2])
            self.assertFalse(np.array_equal(before[3], after[3]))

    def test_model_choice_uses_mae_then_rmse_then_complexity_and_name(self):
        records = [
            {"candidate": "complex", "validation_mae_minutes": 1, "validation_rmse_minutes": 2, "parameter_count": 100},
            {"candidate": "simple", "validation_mae_minutes": 1, "validation_rmse_minutes": 2, "parameter_count": 20},
            {"candidate": "lower_rmse", "validation_mae_minutes": 1, "validation_rmse_minutes": 1.9, "parameter_count": 200},
            {"candidate": "lower_mae", "validation_mae_minutes": .99, "validation_rmse_minutes": 50, "parameter_count": 500},
            {"candidate": "a_name_tie", "validation_mae_minutes": 1, "validation_rmse_minutes": 2, "parameter_count": 20},
        ]
        self.assertEqual([item["candidate"] for item in sorted(records, key=choice_key)],
                         ["lower_mae", "lower_rmse", "a_name_tie", "simple", "complex"])

    def test_default_and_quick_grid_counts_and_neural_training_only_stopping(self):
        grid = candidate_grid(8)
        self.assertEqual(Counter(item[1] for item in grid),
                         {"polynomial": 6, "neural_network": 3, "anfis": 2})
        self.assertEqual(Counter(item[1] for item in candidate_grid(8, quick=True)),
                         {"polynomial": 1, "neural_network": 1, "anfis": 1})
        neural_units = []
        for _, family, parameters, _, model in grid:
            if family == "neural_network":
                neural_units.append(parameters["hidden_units"])
                self.assertFalse(model.regressor.early_stopping)
                self.assertEqual(model.regressor.max_iter, 500)
                self.assertEqual(parameters["stopping"], "training-loss tolerance only")
            elif family == "polynomial":
                self.assertIn(parameters["degree"], [1, 2, 3])
                self.assertIn(parameters["alpha"], [1.0, 10.0])
        self.assertEqual(neural_units, [8, 16, 32])

    def test_quick_training_freezes_all_families_and_reloads_in_fresh_process_without_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            split_dir, features_dir, _, validation = development_fixture(directory)
            encoder_before = (features_dir / "encoder.json").read_bytes()
            output_dir = directory / "step04"
            with patch("preprocessing.pd.read_csv", wraps=pd.read_csv) as reader, \
                 patch("train_models.make_training_plots"), \
                 patch.dict(os.environ, {"LOKY_MAX_CPU_COUNT": "2"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                selection = train_and_select(split_dir, features_dir, output_dir, quick=True)
            self.assertEqual([Path(call.args[0]).name for call in reader.call_args_list],
                             ["train.csv", "validation.csv"])
            self.assertFalse((split_dir / "test.csv").exists())
            self.assertEqual((features_dir / "encoder.json").read_bytes(), encoder_before)
            saved = json.loads((output_dir / "model_selection.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, selection)
            self.assertFalse(saved["fit_on_validation"])
            self.assertIn("not opened", saved["test_status"])
            self.assertEqual(saved["training_rows"], 48)
            self.assertEqual(saved["validation_rows"], 16)
            families = {"median_baseline", "polynomial", "neural_network", "anfis"}
            self.assertEqual({item["family"] for item in saved["selected_models"]}, families)
            self.assertEqual(set(saved["fitted_model_sha256"]), families)
            expected = {}
            sample = validation.iloc[:3].copy()
            sample.to_csv(directory / "prediction_inputs.csv", index=False)
            sample = pd.read_csv(directory / "prediction_inputs.csv")
            for family in families:
                artifact_path = output_dir / f"{family}.joblib"
                self.assertEqual(sha256_file(artifact_path), saved["fitted_model_sha256"][family])
                artifact = joblib.load(artifact_path)
                self.assertEqual(artifact["target_units"], "minutes")
                self.assertEqual(artifact["encoder"]["encoded_columns"], saved["encoded_columns"])
                self.assertEqual(artifact["selected_factors"], saved["selected_factors"])
                if family != "median_baseline":
                    self.assertEqual(int(artifact["model"].transformer_["scale"].n_samples_seen_), 48)
                expected[family] = predict_minutes(artifact["model"], transform_features(sample, artifact["encoder"])).tolist()
            script = (
                "import sys, pathlib, json, joblib, pandas as pd; "
                "sys.path.insert(0,sys.argv[1]); "
                "from preprocessing import transform_features; "
                "from train_models import predict_minutes; "
                "p=pathlib.Path(sys.argv[2]); "
                "frame=pd.read_csv(p/'prediction_inputs.csv'); "
                "result={}; "
                "\nfor f in sorted((p/'step04').glob('*.joblib')):\n"
                " a=joblib.load(f); result[a['family']]=predict_minutes(a['model'],transform_features(frame,a['encoder'])).tolist()\n"
                "(p/'reloaded_predictions.json').write_text(json.dumps(result))"
            )
            completed = subprocess.run([sys.executable, "-c", script, str(SOURCE_DIR), str(directory)],
                                       capture_output=True, text=True, timeout=45)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            observed = json.loads((directory / "reloaded_predictions.json").read_text())
            self.assertEqual(set(observed), families)
            for family in families:
                np.testing.assert_allclose(observed[family], expected[family], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
