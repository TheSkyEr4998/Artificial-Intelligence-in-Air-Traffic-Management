import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing import (
    CONTEXT, DISTANCE_IDENTITY, FEATURES, ROUTE, TARGET,
    encode_values, fit_encoder, fit_route_pca, make_probe, prediction_metrics,
    read_development_data, shortlist_routes, transform_features,
)


def records(count=48, seed=29):
    rng = np.random.default_rng(seed)
    gate = rng.uniform(10, 150, count)
    long = rng.uniform(200, 1400, count)
    other = rng.uniform(20, 600, count)
    return pd.DataFrame({
        "id": [f"flight-{i}" for i in range(count)],
        "source_date": pd.Timestamp("2021-01-01"),
        "type": ["arrival" if i % 2 == 0 else "departure" for i in range(count)],
        TARGET: rng.uniform(2, 30, count),
        "start_hour": np.arange(count) % 24,
        "n_departing_at_start": rng.integers(0, 15, count),
        "n_arriving_at_start": rng.integers(0, 12, count),
        "distance": gate + long + other,
        "angle": rng.uniform(0, 360, count),
        "shortest path": rng.uniform(200, 1800, count),
        "distance_gate": gate,
        "distance_long": long,
        "distance_else": other,
    })


class PreprocessingTests(unittest.TestCase):
    def test_hour_encoding_respects_midnight_proximity(self):
        frame = records(24)
        state = fit_encoder(frame, ["start_hour"])
        hour = frame.iloc[[23, 0, 12]]
        encoded = encode_values(hour, state)
        near = np.linalg.norm(encoded[0] - encoded[1])
        far = np.linalg.norm(encoded[1] - encoded[2])
        self.assertLess(near, far / 5)
        np.testing.assert_allclose(encoded[1], [0, 1], atol=1e-12)
        np.testing.assert_allclose(encoded[2], [0, -1], atol=1e-12)

    def test_type_is_binary_and_targets_and_metadata_do_not_enter_predictors(self):
        frame = records()
        state = fit_encoder(frame, CONTEXT)
        encoded = encode_values(frame, state)
        np.testing.assert_array_equal(encoded[:, 0], np.arange(len(frame)) % 2)
        self.assertEqual(state["encoded_columns"], [
            "is_departure", "start_hour_sin", "start_hour_cos",
            "n_departing_at_start", "n_arriving_at_start",
        ])
        changed = frame.assign(taxi_time=1e9, id="changed", source_date="2040-01-01")
        np.testing.assert_array_equal(transform_features(frame, state), transform_features(changed, state))
        for forbidden in [TARGET, "id", "source_date"]:
            with self.assertRaisesRegex(ValueError, "predictor allowlist"):
                fit_encoder(frame, [forbidden])
        changed.loc[0, "type"] = "vehicle"
        with self.assertRaisesRegex(ValueError, "arrival or departure"):
            transform_features(changed, state)

    def test_imputation_and_scaling_use_training_values_and_remain_frozen(self):
        train = records(12)
        train["distance"] = [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        train["start_hour"] = [2, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        factors = ["type", "start_hour", "distance"]
        state = fit_encoder(train, factors)
        self.assertEqual(state["numeric_medians"]["distance"], 7.0)
        self.assertEqual(state["hour_mode"], 2.0)
        imputed = encode_values(train, state)
        np.testing.assert_allclose(state["mean"], imputed.mean(axis=0))
        np.testing.assert_allclose(state["scale"], imputed.std(axis=0))
        state_before = json.dumps(state, sort_keys=True)
        validation = train.iloc[:2].copy()
        validation.loc[validation.index[0], ["distance", "start_hour"]] = np.nan
        validation.loc[validation.index[1], "distance"] = 1e10
        transformed = transform_features(validation, state)
        expected = (encode_values(validation, state) - state["mean"]) / state["scale"]
        np.testing.assert_allclose(transformed, expected)
        self.assertEqual(json.dumps(state, sort_keys=True), state_before)
        self.assertEqual(state["fit_rows"], len(train))

    def test_route_pca_uses_training_statistics_and_minimal_threshold_components(self):
        train = records()
        train.loc[0, "angle"] = np.nan
        pipeline, z, scores, retained, variance, coverage = fit_route_pca(train, .95)
        np.testing.assert_allclose(pipeline["imputer"].statistics_, train[ROUTE].median().to_numpy())
        filled = train[ROUTE].fillna(train[ROUTE].median()).to_numpy()
        np.testing.assert_allclose(pipeline["scaler"].mean_, filled.mean(axis=0))
        self.assertEqual(pipeline["pca"].n_samples_, len(train))
        self.assertEqual(scores.shape, (len(train), retained))
        self.assertGreaterEqual(variance.cumulative_variance.iloc[retained - 1], .95)
        if retained > 1:
            self.assertLess(variance.cumulative_variance.iloc[retained - 2], .95)
        np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-12)
        self.assertEqual(set(coverage.factor), set(ROUTE))
        mean_before = pipeline["scaler"].mean_.copy()
        components_before = pipeline["pca"].components_.copy()
        pipeline.transform(train[ROUTE].fillna(0) * 1e5)
        np.testing.assert_array_equal(pipeline["scaler"].mean_, mean_before)
        np.testing.assert_array_equal(pipeline["pca"].components_, components_before)

    def test_shortlists_have_three_seven_and_eight_factor_candidates_without_distance_identity(self):
        _, z, scores, _, variance, _ = fit_route_pca(records())
        fraction = float(variance.loc[variance.retained, "variance_ratio"].sum())
        table = shortlist_routes(z, scores, fraction, per_size=3)
        chosen = table.loc[table.shortlisted]
        self.assertEqual(chosen.groupby("factor_count").size().to_dict(), {7: 3, 8: 3})
        self.assertTrue(all(not DISTANCE_IDENTITY.issubset(items) for items in table.route_factors))
        self.assertTrue(all(count == len(items) + len(CONTEXT)
                            for count, items in zip(table.factor_count, table.route_factors)))
        self.assertTrue(table.total_route_variance_loss.ge(-1e-12).all())
        shuffled_order = np.random.default_rng(83).permutation(len(z))
        repeat = shortlist_routes(z[shuffled_order], scores[shuffled_order], fraction, per_size=3)
        self.assertEqual(table.route_factors.tolist(), repeat.route_factors.tolist())
        np.testing.assert_allclose(table.retained_pc_reconstruction_loss,
                                   repeat.retained_pc_reconstruction_loss, atol=1e-12)

    def test_verified_distance_identity_collapses_equivalent_spaces_before_shortlisting(self):
        _, z, scores, _, variance, _ = fit_route_pca(records())
        fraction = float(variance.loc[variance.retained, "variance_ratio"].sum())
        unchecked = shortlist_routes(z, scores, fraction, distance_identity_verified=False)
        self.assertEqual(len(unchecked), 34)
        self.assertTrue(unchecked.equivalent_to.eq("").all())
        table = shortlist_routes(z, scores, fraction, distance_identity_verified=True)
        unique = table.loc[table.equivalent_to.eq("")]
        duplicates = table.loc[table.equivalent_to.ne("")]
        self.assertEqual(len(duplicates), 9)
        self.assertEqual(unique.groupby("factor_count").size().to_dict(), {7: 17, 8: 8})
        self.assertEqual(len(unique), 25)
        self.assertTrue(duplicates.pca_rank_within_size.eq(0).all())
        self.assertFalse(duplicates.shortlisted.any())
        chosen = table.loc[table.shortlisted]
        self.assertEqual(chosen.groupby("factor_count").size().to_dict(), {7: 3, 8: 3})
        signatures = []
        for factors in chosen.route_factors:
            if len(set(factors) & DISTANCE_IDENTITY) == 3:
                signatures.append(("distance_basis_rank3", *[f for f in factors if f not in DISTANCE_IDENTITY]))
            else:
                signatures.append(tuple(factors))
        self.assertEqual(len(signatures), len(set(signatures)))
        expected_representatives = {
            ("distance", "distance_gate", "distance_long"),
            ("distance", "angle", "distance_gate", "distance_long"),
            ("distance", "shortest path", "distance_gate", "distance_long"),
        }
        self.assertEqual({tuple(json.loads(item)) for item in duplicates.equivalent_to},
                         expected_representatives)
        for representative in expected_representatives:
            self.assertEqual(sum(tuple(factors) == representative for factors in unique.route_factors), 1)

    def test_development_reader_succeeds_without_a_test_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            train = records(12)
            validation = records(10, seed=31).assign(source_date=pd.Timestamp("2021-02-01"))
            validation["id"] = [f"validation-{i}" for i in range(len(validation))]
            for name, frame in [("train", train), ("validation", validation)]:
                frame.to_csv(directory / f"{name}.csv", index=False)
            manifest = {
                "candidate_features": FEATURES,
                "partitions": {
                    "train": {"rows": 12, "first_date": "2021-01-01", "last_date": "2021-01-01"},
                    "validation": {"rows": 10, "first_date": "2021-02-01", "last_date": "2021-02-01"},
                },
            }
            (directory / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch("preprocessing.pd.read_csv", wraps=pd.read_csv) as reader:
                observed_train, observed_validation, _ = read_development_data(directory)
            self.assertEqual([Path(call.args[0]).name for call in reader.call_args_list],
                             ["train.csv", "validation.csv"])
            self.assertEqual(len(observed_train), 12)
            self.assertEqual(len(observed_validation), 10)
            self.assertFalse((directory / "test.csv").exists())
            manifest["partitions"]["train"]["rows"] = 999
            (directory / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_development_data(directory)

    def test_encoder_json_and_probe_joblib_survive_a_fresh_python_process(self):
        frame = records(20)
        state = fit_encoder(frame, [*CONTEXT, "distance", "angle", "shortest path"])
        json_state = json.loads(json.dumps(state, allow_nan=False))
        encoded = transform_features(frame, state)
        np.testing.assert_array_equal(transform_features(frame, json_state), encoded)
        model = make_probe().fit(encoded, frame[TARGET])
        expected = model.predict(encoded)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            joblib.dump(model, directory / "probe.joblib")
            np.save(directory / "inputs.npy", encoded)
            script = (
                "import sys, pathlib, joblib, numpy as np; "
                "p=pathlib.Path(sys.argv[1]); "
                "np.save(p/'predictions.npy', joblib.load(p/'probe.joblib').predict(np.load(p/'inputs.npy')))"
            )
            completed = subprocess.run([sys.executable, "-c", script, str(directory)],
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            np.testing.assert_allclose(np.load(directory / "predictions.npy"), expected, atol=1e-12)

    def test_invalid_hour_nonfinite_features_and_predictions_fail_explicitly(self):
        frame = records()
        state = fit_encoder(frame, ["start_hour", "distance"])
        for column, value in [("start_hour", 24), ("start_hour", 2.5), ("distance", np.inf)]:
            with self.subTest(column=column, value=value):
                changed = frame.copy()
                changed[column] = changed[column].astype(float)
                changed.loc[0, column] = value
                with self.assertRaises(ValueError):
                    transform_features(changed, state)
        with self.assertRaisesRegex(ValueError, "nonfinite predictions"):
            prediction_metrics(np.array([1, 2]), np.array([1, np.nan]))
        with self.assertRaisesRegex(ValueError, "nonfinite predictions"):
            prediction_metrics(np.array([1, 2]), np.array([1, np.inf]))
        metrics = prediction_metrics(np.array([1, 2]), np.array([-1, 2]))
        self.assertEqual(metrics["negative_prediction_count"], 1)
        self.assertEqual(metrics["mae_minutes"], 1.0)


if __name__ == "__main__":
    unittest.main()
