"""Trainable, clustered first-order Takagi-Sugeno ANFIS regression.

All fitting uses the rows given to ``fit``; there is no internal holdout. A
bounded training sample initializes K-means only. Every training row contributes
to consequent least squares, premise minibatches, and full-epoch loss. Pass the
same scaled encoded predictors used for the other models, without extra PCA.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class ANFISRegressor(RegressorMixin, BaseEstimator):
    """Gaussian rule premises with affine, first-order Sugeno consequents.

    Hybrid training alternates full-data ridge least squares for consequents
    with Adam minibatch gradients for centres and log widths. Inputs should
    already be scaled. Targets are standardized inside training, and ``predict``
    restores the units passed to ``fit``. When fitted to log1p durations it
    therefore returns log1p durations; inverse transformation belongs to the
    caller's shared evaluation and inference code.

    ``loss_curve_`` records full-training MSE in the supplied target units,
    including initialization and the last epoch. The best training state is
    restored; no validation or test rows are seen here. ``consequents_`` has one
    intercept then one coefficient per encoded predictor, in standardized target
    units. ``centers_`` and ``sigmas_`` use the supplied encoded-input units.
    """

    def __init__(
        self,
        n_rules: int = 4,
        epochs: int = 25,
        learning_rate: float = 0.01,
        ridge: float = 0.002,
        random_state: int = 42,
        batch_size: int = 4096,
        initialization_max_samples: int = 5000,
        min_sigma: float = 0.15,
        max_sigma: float = 5.0,
    ):
        self.n_rules = n_rules
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.ridge = ridge
        self.random_state = random_state
        self.batch_size = batch_size
        self.initialization_max_samples = initialization_max_samples
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    @property
    def sigmas_(self):
        """Readable learned Gaussian widths; positivity follows from log widths."""
        check_is_fitted(self, "log_sigmas_")
        return np.exp(self.log_sigmas_)

    def _validate_parameters(self):
        for name, minimum in (("n_rules", 1), ("epochs", 15), ("batch_size", 1),
                              ("initialization_max_samples", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.initialization_max_samples < self.n_rules:
            raise ValueError("initialization_max_samples must be >= n_rules")
        for name in ("learning_rate", "ridge", "min_sigma", "max_sigma"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")
        if self.min_sigma > self.max_sigma:
            raise ValueError("min_sigma must be <= max_sigma")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, (int, np.integer)) or self.random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")

    def _weights(self, X):
        differences = X[:, None, :] - self.centers_[None, :, :]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                scaled = differences / np.exp(self.log_sigmas_)[None, :, :]
                log_firing = -0.5 * np.sum(scaled**2, axis=2)
                # Stabilize product memberships as a log-softmax. Underflow of
                # weaker rules is harmless; at least the strongest stays one.
                log_firing -= log_firing.max(axis=1, keepdims=True)
                firing = np.exp(log_firing)
                weights = firing / firing.sum(axis=1, keepdims=True)
            except FloatingPointError as exc:
                raise ValueError("ANFIS membership calculation overflowed; check scaled input magnitudes") from exc
        return weights, differences

    def _forward(self, X):
        weights, differences = self._weights(X)
        augmented = np.column_stack((np.ones(len(X)), X))
        rule_output = augmented @ self.consequents_.T
        prediction = np.sum(weights * rule_output, axis=1)
        if not np.isfinite(prediction).all():
            raise ValueError("ANFIS produced non-finite outputs during training or prediction")
        return prediction, weights, differences, rule_output

    def _solve_consequents(self, X, target):
        """Accumulate full-training normal equations in bounded-memory batches."""
        coefficients = self.n_rules_ * (self.n_features_in_ + 1)
        gram = np.zeros((coefficients, coefficients), dtype=float)
        rhs = np.zeros(coefficients, dtype=float)
        for start in range(0, len(X), self.batch_size):
            batch = X[start:start + self.batch_size]
            weights, _ = self._weights(batch)
            augmented = np.column_stack((np.ones(len(batch)), batch))
            design = (weights[:, :, None] * augmented[:, None, :]).reshape(len(batch), -1)
            gram += design.T @ design
            rhs += design.T @ target[start:start + self.batch_size]
        gram /= len(X)
        rhs /= len(X)
        # Normalization makes regularization strength independent of data size.
        # Penalize all coefficients, including poorly supported rule intercepts.
        gram.flat[:: coefficients + 1] += self.ridge
        if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
            raise ValueError("ANFIS least-squares equations are non-finite; check input scaling")
        try:
            solution = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError("ANFIS ridge consequent system could not be solved") from exc
        if not np.isfinite(solution).all():
            raise ValueError("ANFIS consequent coefficients are non-finite")
        self.consequents_ = solution.reshape(self.n_rules_, self.n_features_in_ + 1)

    def _premise_gradients(self, X, target):
        """Analytic MSE gradients while consequent coefficients are held fixed."""
        prediction, weights, differences, rule_output = self._forward(X)
        gradient_log_firing = (
            2 / len(X) * (prediction - target)[:, None]
            * weights * (rule_output - prediction[:, None])
        )
        inverse_variance = np.exp(-2 * self.log_sigmas_)
        gradient_centres = np.sum(
            gradient_log_firing[:, :, None] * differences * inverse_variance[None, :, :], axis=0
        )
        gradient_widths = np.sum(
            gradient_log_firing[:, :, None] * differences**2 * inverse_variance[None, :, :], axis=0
        )
        if not np.isfinite(gradient_centres).all() or not np.isfinite(gradient_widths).all():
            raise ValueError("ANFIS premise gradients diverged; check target and input scales")
        return gradient_centres, gradient_widths

    def _training_mse(self, X, target):
        squared_error = 0.0
        for start in range(0, len(X), self.batch_size):
            prediction, _, _, _ = self._forward(X[start:start + self.batch_size])
            squared_error += float(np.sum((prediction - target[start:start + self.batch_size]) ** 2))
        mse = squared_error / len(X)
        if not np.isfinite(mse):
            raise ValueError("ANFIS training loss diverged")
        return mse

    def fit(self, X, y):
        self._validate_parameters()
        X, y = check_X_y(X, y, dtype=np.float64)
        if len(X) < 2:
            raise ValueError("ANFIS needs at least two training examples")
        self.n_features_in_ = X.shape[1]
        self.n_training_samples_ = len(X)
        self.target_mean_ = float(np.mean(y))
        self.target_scale_ = max(float(np.std(y)), 1e-12)
        if not np.isfinite(self.target_mean_) or not np.isfinite(self.target_scale_):
            raise ValueError("ANFIS target standardization overflowed")
        target = (y - self.target_mean_) / self.target_scale_
        random = np.random.default_rng(self.random_state)
        n_initialization = min(len(X), self.initialization_max_samples)
        initialization_indices = random.choice(len(X), n_initialization, replace=False)
        initialization = X[initialization_indices]
        self.n_initialization_samples_ = n_initialization
        self.n_rules_ = min(self.n_rules, len(np.unique(initialization, axis=0)))
        clustering = KMeans(n_clusters=self.n_rules_, n_init=5, random_state=self.random_state)
        clustering.fit(initialization)
        groups = clustering.predict(X)
        self.centers_ = clustering.cluster_centers_.copy()
        global_width = np.maximum(X.std(axis=0), self.min_sigma)
        widths = np.empty_like(self.centers_)
        for rule in range(self.n_rules_):
            members = X[groups == rule]
            within = members.std(axis=0) if len(members) > 1 else global_width
            widths[rule] = np.maximum(within, 0.5 * global_width)
        self.log_sigmas_ = np.log(np.clip(widths, self.min_sigma, self.max_sigma))
        self.initial_centers_ = self.centers_.copy()
        self.initial_log_sigmas_ = self.log_sigmas_.copy()
        self.loss_curve_ = []
        self.best_epoch_ = 0
        self._solve_consequents(X, target)
        best_loss = self._training_mse(X, target)
        self.loss_curve_.append(best_loss * self.target_scale_**2)
        best_state = (self.centers_.copy(), self.log_sigmas_.copy(), self.consequents_.copy())
        moments = [np.zeros_like(self.centers_), np.zeros_like(self.log_sigmas_)]
        variances = [np.zeros_like(self.centers_), np.zeros_like(self.log_sigmas_)]
        updates = 0
        for epoch in range(self.epochs):
            # Every row participates once per epoch; no fitting subset is taken.
            order = random.permutation(len(X))
            for start in range(0, len(X), self.batch_size):
                indices = order[start:start + self.batch_size]
                gradients = self._premise_gradients(X[indices], target[indices])
                updates += 1
                for gradient, moment, variance, parameter in zip(
                    gradients, moments, variances, (self.centers_, self.log_sigmas_)
                ):
                    gradient = np.clip(gradient, -5, 5)
                    moment *= 0.9
                    moment += 0.1 * gradient
                    variance *= 0.999
                    variance += 0.001 * gradient**2
                    corrected_moment = moment / (1 - 0.9**updates)
                    corrected_variance = variance / (1 - 0.999**updates)
                    parameter -= self.learning_rate * corrected_moment / (np.sqrt(corrected_variance) + 1e-8)
                np.clip(self.log_sigmas_, np.log(self.min_sigma), np.log(self.max_sigma), out=self.log_sigmas_)
                if not np.isfinite(self.centers_).all():
                    raise ValueError("ANFIS Gaussian centres diverged")
            self._solve_consequents(X, target)
            loss = self._training_mse(X, target)
            self.loss_curve_.append(loss * self.target_scale_**2)
            if loss < best_loss:
                best_loss = loss
                self.best_epoch_ = epoch + 1
                best_state = (self.centers_.copy(), self.log_sigmas_.copy(), self.consequents_.copy())
        self.centers_, self.log_sigmas_, self.consequents_ = best_state
        self.n_iter_ = self.epochs
        self.n_premise_updates_ = updates
        self.best_training_mse_ = best_loss * self.target_scale_**2
        return self

    def predict(self, X):
        check_is_fitted(self, "consequents_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("ANFIS predictor count differs from its fitted encoded-input schema")
        predictions = []
        for start in range(0, len(X), self.batch_size):
            normalized, _, _, _ = self._forward(X[start:start + self.batch_size])
            predictions.append(normalized * self.target_scale_ + self.target_mean_)
        prediction = np.concatenate(predictions)
        if not np.isfinite(prediction).all():
            raise ValueError("ANFIS predictions are non-finite after restoring target units")
        return prediction
