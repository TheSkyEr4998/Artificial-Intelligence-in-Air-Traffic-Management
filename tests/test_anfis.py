import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
from anfis import ANFISRegressor


class ANFISTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"LOKY_MAX_CPU_COUNT": "2"}))
        self.enterContext(threadpool_limits(limits=2))

    def test_hybrid_fit_changes_premises_and_beats_constant_median(self):
        random = np.random.default_rng(17)
        X = random.uniform(-2, 2, (240, 2))
        y = 5 + np.sin(1.7 * X[:, 0]) + 0.6 * X[:, 1]**2
        model = ANFISRegressor(n_rules=4, epochs=25, batch_size=80).fit(X, y)
        independent = random.uniform(-2, 2, (100, 2))
        actual = 5 + np.sin(1.7 * independent[:, 0]) + 0.6 * independent[:, 1]**2
        error = np.mean(np.abs(actual - model.predict(independent)))
        baseline_error = np.mean(np.abs(actual - np.median(y)))
        self.assertLess(error, baseline_error * 0.65)
        self.assertGreater(np.max(np.abs(model.centers_ - model.initial_centers_)), 1e-5)
        self.assertGreater(np.max(np.abs(model.log_sigmas_ - model.initial_log_sigmas_)), 1e-5)
        self.assertLess(model.best_training_mse_, model.loss_curve_[0])
        self.assertEqual(len(model.loss_curve_), model.epochs + 1)
        self.assertTrue(np.all(model.sigmas_ >= model.min_sigma - 1e-12))
        self.assertTrue(np.all(model.sigmas_ <= model.max_sigma + 1e-12))

    def test_analytic_premise_gradients_match_finite_differences(self):
        # Check the normalized rule-weight chain rule independently by varying
        # parameters in forward predictions with fixed affine consequents.
        model = ANFISRegressor(n_rules=2)
        model.centers_ = np.array([[-0.4, 0.2], [0.5, -0.3]])
        model.log_sigmas_ = np.log(np.array([[0.8, 1.2], [0.9, 0.7]]))
        model.consequents_ = np.array([[0.1, 0.4, -0.3], [-0.2, 0.8, 0.5]])
        X = np.array([[-0.2, 0.4], [0.3, -0.5], [0.7, 0.2]])
        target = np.array([0.6, -0.1, 0.3])
        gradients = model._premise_gradients(X, target)
        epsilon = 1e-6
        for parameter, analytic in zip((model.centers_, model.log_sigmas_), gradients):
            numerical = np.empty_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                upper = np.mean((model._forward(X)[0] - target)**2)
                parameter[index] = original - epsilon
                lower = np.mean((model._forward(X)[0] - target)**2)
                parameter[index] = original
                numerical[index] = (upper - lower) / (2 * epsilon)
            np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-7)

    def test_only_initialization_is_sampled_and_fit_is_deterministic(self):
        random = np.random.default_rng(7)
        X = random.normal(size=(210, 3))
        target = X[:, 0] + np.sin(X[:, 1])
        initial = ANFISRegressor(epochs=15, batch_size=70, initialization_max_samples=50)
        first = clone(initial).fit(X, target)
        second = clone(initial).fit(X, target)
        self.assertEqual(first.n_initialization_samples_, 50)
        self.assertEqual(first.n_training_samples_, 210)
        self.assertEqual(first.n_premise_updates_, 45)
        np.testing.assert_allclose(first.predict(X), second.predict(X), atol=1e-12)
        np.testing.assert_allclose(first.centers_, second.centers_, atol=1e-12)
        self.assertEqual(first.n_features_in_, 3)

    def test_far_inputs_are_finite_and_widths_stay_positive(self):
        random = np.random.default_rng(9)
        X = random.normal(size=(100, 2))
        model = ANFISRegressor(epochs=15).fit(X, np.sin(X[:, 0]))
        far = np.array([[1000, -1000], [-1000, 1000]])
        self.assertTrue(np.isfinite(model.predict(far)).all())
        weights, _ = model._weights(far)
        np.testing.assert_allclose(weights.sum(axis=1), [1, 1])
        self.assertTrue((model.sigmas_ > 0).all())
        with self.assertRaisesRegex(ValueError, "predictor count"):
            model.predict(np.ones((3, 3)))

    def test_artifact_loads_in_a_fresh_python_process(self):
        random = np.random.default_rng(13)
        X = random.normal(size=(80, 2))
        model = ANFISRegressor(epochs=15).fit(X, np.log1p(3 + X[:, 0]**2))
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            joblib.dump(model, destination / "model.joblib")
            np.savez(destination / "check.npz", X=X, expected=model.predict(X))
            script = (
                "import joblib, numpy as np, sys; "
                "model=joblib.load(sys.argv[1]); data=np.load(sys.argv[2]); "
                "np.testing.assert_allclose(model.predict(data['X']), data['expected'], rtol=1e-12); "
                "print('fresh-process prediction parity')"
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(SOURCE) + os.pathsep + environment.get("PYTHONPATH", "")
            result = subprocess.run(
                [sys.executable, "-c", script, str(destination / "model.joblib"), str(destination / "check.npz")],
                env=environment, text=True, capture_output=True, timeout=30, check=True,
            )
            self.assertIn("fresh-process prediction parity", result.stdout)

    def test_invalid_parameters_and_nonfinite_values_fail_explicitly(self):
        X = np.ones((5, 2))
        target = np.arange(5, dtype=float)
        for parameters in ({"epochs": 14}, {"n_rules": 0}, {"batch_size": 0},
                           {"ridge": 0}, {"learning_rate": float("nan")},
                           {"min_sigma": 2, "max_sigma": 1}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                ANFISRegressor(**parameters).fit(X, target)
        with self.assertRaises(ValueError):
            ANFISRegressor().fit(X, [1, 2, 3, 4, float("inf")])


if __name__ == "__main__":
    unittest.main()
