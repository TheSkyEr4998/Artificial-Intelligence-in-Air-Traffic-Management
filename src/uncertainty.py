"""Step 5 uncertainty for frozen models, resampling whole source dates.

All families share exactly the same day draws. Resampling individual records
would discard dependence among journeys on the same source date. These
percentile intervals are conditional on the fitted models and supplied test
records. They assume independence between source-date groups and do not account
for training, feature-selection, label, or source-date assignment uncertainty.
They are neither posterior probabilities nor p-values, and must not guide
further model selection on this test set.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _interval(point, replicates):
    """Use only defined metric replicates, exposing every excluded replicate."""
    values = np.asarray(replicates, dtype=float)
    valid = values[np.isfinite(values)]
    bounds = np.quantile(valid, [.025, .975]) if len(valid) else [None, None]
    return {
        "point": float(point) if point is not None and np.isfinite(point) else None,
        "lower95": float(bounds[0]) if bounds[0] is not None else None,
        "upper95": float(bounds[1]) if bounds[1] is not None else None,
        "valid_replicates": int(len(valid)),
        "invalid_replicates": int(len(values) - len(valid)),
    }


def cluster_bootstrap_metrics(
    dates, y, predictions, winner, iterations=2000, seed=42,
):
    """Return JSON-safe scores, intervals, and paired winner-minus-family MAE.

    ``dates`` is a row-aligned source_date Series (ISO calendar-date strings or
    datetime/date values), ``y`` is a finite one-dimensional target vector, and ``predictions``
    maps each family name to its finite one-dimensional prediction vector. Every
    vector must use the same record order. The winner must already have been
    frozen using development data; this function fits and chooses nothing.

    A replicate samples the original number of whole source dates with
    replacement. It retains every record from each selected day, repeating that
    day's records when drawn repeatedly. Sufficient per-day sums avoid creating
    copies of the thousands of test rows. MAE and RMSE weight individual records,
    rather than assigning equal weight to daily average errors. R² uses the
    actual target's mean within each replicate, with no force-finite replacement.

    At least two distinct source dates are required. Source_date is a grouping
    proxy, not proof of precise event timing. Replicates with zero or numerically
    indistinguishable target variance have undefined R²: their NaNs are excluded
    from R² quantiles, counts are reported, and bounds are None if none are
    defined. When any R² replicate is undefined, its reported interval describes
    the defined replicates only. Differences use winner MAE minus comparator MAE;
    negative values indicate lower observed error, without a significance claim.
    """
    if (not isinstance(iterations, (int, np.integer)) or isinstance(iterations, bool)
            or iterations < 1):
        raise ValueError("iterations must be a positive integer")
    if (not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(predictions, Mapping) or not predictions:
        raise ValueError("predictions must be a nonempty mapping of family names to vectors")
    if any(not isinstance(name, str) or not name.strip() for name in predictions):
        raise ValueError("Prediction family keys must be nonempty strings")
    if not isinstance(winner, str) or winner not in predictions:
        raise ValueError("winner must name one of the provided prediction families")
    target = np.asarray(y, dtype=float)
    if target.ndim != 1 or not len(target) or not np.isfinite(target).all():
        raise ValueError("y must be a nonempty finite one-dimensional vector")
    source_dates = pd.Series(dates).reset_index(drop=True)
    if len(source_dates) != len(target):
        raise ValueError("source dates and targets must have the same row count")
    if any(isinstance(value, (bool, int, float, np.integer, np.floating)) for value in source_dates):
        raise ValueError("source_date must contain calendar dates, not numeric timestamp keys")
    try:
        parsed_dates = pd.to_datetime(source_dates, format="ISO8601", errors="raise")
        if parsed_dates.isna().any():
            raise ValueError("missing source date")
        day_keys = parsed_dates.dt.strftime("%Y-%m-%d").to_numpy()
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("source_date must contain valid, nonmissing calendar dates") from exc
    days, group_codes = np.unique(day_keys, return_inverse=True)
    n_days = len(days)
    if n_days < 2:
        raise ValueError("At least two distinct source dates are required for day-cluster uncertainty")
    families = sorted(predictions)
    matrix = []
    for family in families:
        values = np.asarray(predictions[family], dtype=float)
        if values.ndim != 1 or len(values) != len(target) or not np.isfinite(values).all():
            raise ValueError(f"{family} predictions must be a finite vector matching the target row count")
        matrix.append(values)
    matrix = np.asarray(matrix)
    counts = np.bincount(group_codes, minlength=n_days).astype(float)

    # Center target moments on a fixed test-set origin for numerical stability.
    # The bootstrap variance still subtracts each replicate's own mean offset.
    with np.errstate(over="ignore", invalid="ignore"):
        center = np.mean(target)
        offsets = target - center
        sum_y = np.bincount(group_codes, weights=offsets, minlength=n_days)
        sum_y_squared = np.bincount(group_codes, weights=offsets ** 2, minlength=n_days)
        errors = matrix - target[None, :]
        sum_absolute_error = np.asarray([
            np.bincount(group_codes, weights=np.abs(error), minlength=n_days) for error in errors
        ])
        sum_squared_error = np.asarray([
            np.bincount(group_codes, weights=error ** 2, minlength=n_days) for error in errors
        ])
    if not all(np.isfinite(values).all() for values in
               [sum_y, sum_y_squared, sum_absolute_error, sum_squared_error]):
        raise ValueError("Target or prediction magnitudes exceed finite metric arithmetic")

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_days, size=(iterations, n_days))
    multiplicity = np.zeros((iterations, n_days), dtype=np.int32)
    np.add.at(multiplicity, (np.arange(iterations)[:, None], draws), 1)
    replicate_counts = multiplicity @ counts
    replicate_sae = sum_absolute_error @ multiplicity.T
    replicate_sse = sum_squared_error @ multiplicity.T
    replicate_mae = replicate_sae / replicate_counts[None, :]
    replicate_rmse = np.sqrt(replicate_sse / replicate_counts[None, :])
    replicate_sum_y = multiplicity @ sum_y
    replicate_sum_y_squared = multiplicity @ sum_y_squared
    mean_correction = replicate_sum_y ** 2 / replicate_counts
    variance = replicate_sum_y_squared - mean_correction
    # Roundoff can leave a tiny positive/negative residue when sampled targets
    # are constant. Do not turn that into a finite, arbitrarily extreme R².
    tolerance = (64 * np.finfo(float).eps
                 * np.maximum(np.abs(replicate_sum_y_squared), np.abs(mean_correction)))
    variance_defined = variance > tolerance
    replicate_r2 = np.full_like(replicate_sse, np.nan, dtype=float)
    replicate_r2[:, variance_defined] = (
        1 - replicate_sse[:, variance_defined] / variance[None, variance_defined]
    )

    observed_mae = sum_absolute_error.sum(axis=1) / len(target)
    observed_rmse = np.sqrt(sum_squared_error.sum(axis=1) / len(target))
    observed_sum_y_squared = float(sum_y_squared.sum())
    observed_mean_correction = float(sum_y.sum() ** 2 / len(target))
    observed_variance = observed_sum_y_squared - observed_mean_correction
    observed_tolerance = (64 * np.finfo(float).eps
                          * max(abs(observed_sum_y_squared), abs(observed_mean_correction)))
    observed_r2 = (1 - sum_squared_error.sum(axis=1) / observed_variance
                   if observed_variance > observed_tolerance else np.full(len(families), np.nan))
    models = {}
    for index, family in enumerate(families):
        models[family] = {
            "mae_minutes": _interval(observed_mae[index], replicate_mae[index]),
            "rmse_minutes": _interval(observed_rmse[index], replicate_rmse[index]),
            "r2": _interval(observed_r2[index], replicate_r2[index]),
        }
        models[family]["r2"]["interval_population"] = (
            "defined_replicates_only" if not variance_defined.all() else "all_replicates"
        )
    winner_index = families.index(winner)
    paired = {}
    for index, comparator in enumerate(families):
        if comparator == winner:
            continue
        paired[comparator] = {
            "winner": winner, "comparator": comparator,
            **_interval(observed_mae[winner_index] - observed_mae[index],
                        replicate_mae[winner_index] - replicate_mae[index]),
        }
    return {
        "method": "paired whole-source-date cluster bootstrap with percentile intervals",
        "confidence_level": .95,
        "iterations": int(iterations), "seed": int(seed),
        "n_rows": int(len(target)), "n_days": int(n_days),
        "source_dates": days.tolist(), "winner": winner,
        "difference_definition": "winner MAE minus comparator MAE; negative means lower winner error",
        "metric_units": {"mae_minutes": "minutes", "rmse_minutes": "minutes", "r2": "dimensionless",
                         "paired_mae_differences": "minutes"},
        "models": models, "paired_mae_differences": paired,
        "limitations": [
            "Conditional on frozen fitted models and these observed test records",
            "Whole-day resampling preserves within-source-date dependence but assumes independent days",
            "Source dates are a proxy for event-day grouping; exact event timing is not established",
            "Does not include training, feature-selection, heuristic-label, or date-assignment uncertainty",
            "Undefined R2 replicates are counted and excluded; its bounds then describe defined replicates only",
            "Intervals and error differences are not probabilities, p-values, or proof of superiority",
        ],
    }
