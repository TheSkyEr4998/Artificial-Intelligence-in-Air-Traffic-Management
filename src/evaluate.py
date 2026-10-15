"""Step 5: evaluate frozen models, without fitting or choosing from test scores."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from create_splits import FEATURES, TARGET, modelling_table, sha256_file
from model_io import load_frozen_models, predict_from_artifact, validate_prediction_inputs
from preprocessing import transform_features
from uncertainty import cluster_bootstrap_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def verify_development_provenance(split_dir, features_dir, selection):
    contract_path = features_dir / "selected_features.json"
    if sha256_file(contract_path) != selection["step3_contract_sha256"] or sha256_file(features_dir / "encoder.json") != selection["encoder_sha256"]:
        raise ValueError("Step 3 feature/encoding contract changed since model training")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for name, digest in [("train.csv", selection["train_sha256"]),
                         ("validation.csv", selection["validation_sha256"]),
                         ("split_manifest.json", contract["split_manifest_sha256"])]:
        if sha256_file(split_dir / name) != digest:
            raise ValueError(f"Development provenance changed: {name}")
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest != selection["split_manifest"]:
        raise ValueError("Step 2 split no longer matches the frozen training run")
    return manifest


def read_verified_test(split_dir, prepared_path, manifest, factors):
    """Match the test partition to its original prepared-source population."""
    test = pd.read_csv(split_dir / "test.csv", dtype={"id": "string", "type": "string"})
    required = ["id", "type", "source_date", TARGET, *[f for f in FEATURES if f != "type"]]
    if not set(required).issubset(test.columns):
        raise ValueError("Test partition columns do not match Step 2")
    test["source_date"] = pd.to_datetime(test["source_date"], format="ISO8601", errors="coerce")
    if test.empty or test["source_date"].isna().any() or test["id"].isna().any() or test.duplicated(["id", "type"]).any():
        raise ValueError("Invalid test metadata or duplicate record keys")
    test[TARGET] = pd.to_numeric(test[TARGET], errors="raise")
    if not (np.isfinite(test[TARGET]) & test[TARGET].gt(0)).all():
        raise ValueError("Test targets must be finite and positive")
    test = validate_prediction_inputs(test, factors)
    details = manifest["partitions"]["test"]
    if len(test) != details["rows"] or test.source_date.nunique() != details["date_groups"] or test.source_date.min().strftime("%Y-%m-%d") != details["first_date"] or test.source_date.max().strftime("%Y-%m-%d") != details["last_date"]:
        raise ValueError("Test dates/counts disagree with the frozen split")
    prepared = pd.read_csv(prepared_path, dtype={"id": "string", "type": "string"})
    original, _, _ = modelling_table(prepared)  # Rule validation only; no learned fit.
    boundary = pd.Timestamp(manifest["validation_cutoff"]) + pd.Timedelta(days=manifest["gap_days"])
    expected = original.loc[original.source_date.gt(boundary)].sort_values(["source_date", "id", "type"]).reset_index(drop=True)
    observed = test.sort_values(["source_date", "id", "type"]).reset_index(drop=True)
    keys = ["id", "type", "source_date"]
    if len(expected) != len(observed) or not expected[keys].equals(observed[keys]):
        raise ValueError("Test record keys no longer match the prepared source")
    numeric = [TARGET, *[f for f in FEATURES if f != "type"]]
    if not np.allclose(expected[numeric].to_numpy(dtype=float), observed[numeric].to_numpy(dtype=float), rtol=1e-10, atol=1e-8, equal_nan=True):
        raise ValueError("Test labels or predictors no longer match the prepared source")
    return test


def heldout_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape or not len(actual) or not np.isfinite(actual).all() or not np.isfinite(predicted).all() or np.any(actual <= 0):
        raise ValueError("Invalid held-out target/prediction arrays")
    error = predicted - actual
    absolute = np.abs(error)
    mae, rmse = float(np.mean(absolute)), float(np.sqrt(np.mean(error**2)))
    if not np.isfinite(mae) or not np.isfinite(rmse):
        raise ValueError("Held-out error calculation overflowed")
    r2 = None if len(actual) < 2 or np.ptp(actual) == 0 else float(1 - np.sum(error**2) / np.sum((actual - actual.mean())**2))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative = absolute / actual * 100
    return {"rows": len(actual), "mae_minutes": mae, "rmse_minutes": rmse, "r2": r2,
            "median_absolute_error_minutes": float(np.median(absolute)),
            "bias_minutes_predicted_minus_actual": float(np.mean(error)),
            "mean_relative_error_percent": float(relative.mean()) if np.isfinite(relative).all() else None,
            "nonfinite_relative_error_terms": int((~np.isfinite(relative)).sum()),
            "fraction_within_2_minutes": float(np.mean(absolute <= 2)),
            "fraction_within_5_minutes": float(np.mean(absolute <= 5)),
            "negative_prediction_count": int(np.sum(predicted < 0))}


def make_test_plots(test, predictions, winner, metrics, intervals, output_dir):
    names = list(predictions)
    labels = {"anfis": "ANFIS", "neural_network": "Neural network", "polynomial": "Polynomial", "median_baseline": "Median baseline"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, measure, title in zip(axes, ["mae_minutes", "rmse_minutes"], ["MAE", "RMSE"]):
        points = [metrics[name][measure] for name in names]
        positions = np.arange(len(names))
        ax.bar(positions, points, color=["#237a91" if name == winner else "#97bbc4" for name in names])
        for x, name in zip(positions, names):
            interval = intervals["models"][name][measure]
            ax.vlines(x, interval["lower95"], interval["upper95"], color="#333333")
            ax.hlines([interval["lower95"], interval["upper95"]], x-.09, x+.09, color="#333333")
        ax.set_xticks(positions, [labels[n] for n in names], rotation=20, ha="right")
        ax.set(ylabel=f"{title} (minutes)", title=f"Test {title}: 95% source-date bootstrap intervals")
    fig.savefig(output_dir / "test_model_comparison.png", dpi=160)
    plt.close(fig)
    actual, chosen = test[TARGET].to_numpy(), predictions[winner]
    fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
    ax.scatter(actual, chosen, s=10, alpha=.25, color="#237a91", edgecolors="none")
    low, high = min(0.0, float(chosen.min())), max(float(actual.max()), float(chosen.max()))
    ax.plot([low, high], [low, high], "--", color="#333333", label="Exact prediction")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set(xlabel="Recorded taxi time (minutes)", ylabel="Predicted taxi time (minutes)",
           title=f"Validation-selected {labels[winner]}: all {len(test):,} test records")
    ax.legend()
    fig.savefig(output_dir / "winner_test_scatter.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.scatter(chosen, actual - chosen, s=10, alpha=.25, color="#237a91", edgecolors="none")
    ax.axhline(0, color="#333333", linestyle="--")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set(xlabel="Predicted taxi time (minutes)", ylabel="Actual minus predicted (minutes)",
           title="Winner residuals: all records, symlog axes retain signs and extremes")
    fig.savefig(output_dir / "winner_test_residuals.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for name in names:
        errors = np.sort(np.abs(predictions[name] - actual))
        ax.step(errors, np.arange(1, len(errors)+1) / len(errors), where="post", label=labels[name])
    ax.set_xscale("symlog", linthresh=1)
    ax.set(xlabel="Absolute prediction error (minutes)", ylabel="Fraction of test records at or below error",
           ylim=(0, 1.02), title="Full-cohort test error distributions")
    ax.legend()
    fig.savefig(output_dir / "test_absolute_error_cdf.png", dpi=160)
    plt.close(fig)


def evaluate_models(split_dir, features_dir, model_dir, output_dir, prepared_path=None, iterations=2000, seed=42):
    split_dir, features_dir, model_dir, output_dir = map(Path, [split_dir, features_dir, model_dir, output_dir])
    prepared_path = Path(prepared_path) if prepared_path else split_dir.parent / "step01" / "features_prepared.csv"
    selection, artifacts = load_frozen_models(model_dir)
    manifest = verify_development_provenance(split_dir, features_dir, selection)
    if sha256_file(prepared_path) != manifest["input_sha256"]:
        raise ValueError("Prepared source changed since the Step 2 split")
    # The selected winner is fixed here, before loading test labels.
    winner = selection["winner_family"]
    configuration = {
        "protocol_version": 1, "selection_sha256": sha256_file(model_dir / "model_selection.json"),
        "test_sha256": sha256_file(split_dir / "test.csv"), "prepared_sha256": sha256_file(prepared_path),
        "model_sha256": selection["fitted_model_sha256"], "winner_family": winner,
        "bootstrap_iterations": iterations, "bootstrap_seed": seed,
        "evaluation_source_sha256": {name: sha256_file(Path(__file__).resolve().parent / name)
                                     for name in ["evaluate.py", "model_io.py", "uncertainty.py", "create_splits.py"]},
    }
    completed = output_dir / "evaluation_manifest.json"
    if completed.exists():
        previous = json.loads(completed.read_text(encoding="utf-8"))
        if previous["configuration"] != configuration:
            raise ValueError("Existing final evaluation belongs to a different build/input/configuration; preserve it and avoid treating a reused test block as untouched")
        for name, digest in previous["output_sha256"].items():
            if sha256_file(output_dir / name) != digest:
                raise ValueError(f"Completed evaluation output changed: {name}")
        print(f"Identical evaluation already completed; reusing saved results in {output_dir}")
        return json.loads((output_dir / "test_results.json").read_text(encoding="utf-8"))
    test = read_verified_test(split_dir, prepared_path, manifest, selection["selected_factors"])
    with threadpool_limits(limits=2):
        predictions = {name: predict_from_artifact(test, artifact) for name, artifact in artifacts.items()}
    actual = test[TARGET].to_numpy()
    metrics = {name: heldout_metrics(actual, prediction) for name, prediction in predictions.items()}
    intervals = cluster_bootstrap_metrics(test.source_date, actual, predictions, winner, iterations=iterations, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"family": name, **score} for name, score in metrics.items()]).to_csv(output_dir / "test_metrics.csv", index=False)
    tables, groups = [], []
    for name, prediction in predictions.items():
        table = test[["id", "type", "source_date"]].copy()
        table["family"] = name
        table["actual_minutes"], table["predicted_minutes"] = actual, prediction
        table["residual_minutes"] = actual - prediction
        table["absolute_error_minutes"] = np.abs(actual - prediction)
        tables.append(table)
        for direction in ["arrival", "departure"]:
            mask = test.type.eq(direction).to_numpy()
            if mask.any():
                groups.append({"family": name, "group_type": "movement", "group": direction,
                               **heldout_metrics(actual[mask], prediction[mask])})
        for band, mask in [("under_1_minute", actual < 1), ("1_to_120_minutes", (actual >= 1) & (actual <= 120)),
                           ("over_120_minutes", actual > 120)]:
            if mask.any():
                groups.append({"family": name, "group_type": "recorded_duration_band", "group": band,
                               **heldout_metrics(actual[mask], prediction[mask])})
    pd.concat(tables, ignore_index=True).to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(groups).to_csv(output_dir / "test_group_diagnostics.csv", index=False)
    test[["id", "source_date", TARGET, *selection["selected_factors"]]].to_csv(output_dir / "test_selected.csv", index=False)
    encoded = pd.DataFrame(transform_features(test, artifacts[winner]["encoder"]), columns=selection["encoded_columns"])
    encoded.insert(0, TARGET, actual)
    encoded.to_csv(output_dir / "test_encoded.csv", index=False)
    (output_dir / "confidence_intervals.json").write_text(json.dumps(intervals, indent=2, allow_nan=False), encoding="utf-8")
    result = {
        "winner_selected_on_validation": winner, "winner_candidate": selection["winner_candidate"],
        "test_used_for_model_selection": False, "refit_during_evaluation": False,
        "quick_training_budget": selection["quick_training_budget"], "test_rows": len(test),
        "test_days": int(test.source_date.nunique()), "first_test_date": test.source_date.min().strftime("%Y-%m-%d"),
        "last_test_date": test.source_date.max().strftime("%Y-%m-%d"), "metrics": metrics,
        "notes": [
            "Validation-selected winner is retained regardless of the descriptive test ranking",
            "All full-cohort positive labels and all raw predictions are scored; no duration filter or prediction clipping",
            "Duration-band diagnostics are descriptive only and were not used to tune cleaning, features or models",
            "MRE can be dominated by short recorded durations; MAE/RMSE remain primary error measures",
            f"Intervals resample whole source dates jointly; {int(test.source_date.nunique())} dates are available in this test block",
            "Day independence and representativeness remain assumptions; seasonal coverage and serial dependence are limited",
            "Accuracy is against inferred completed-journey labels, not official verified live taxi-time forecasts",
        ],
    }
    (output_dir / "test_results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    make_test_plots(test, predictions, winner, metrics, intervals, output_dir)
    files = ["test_metrics.csv", "test_predictions.csv", "test_group_diagnostics.csv", "test_selected.csv", "test_encoded.csv",
             "confidence_intervals.json", "test_results.json", "test_model_comparison.png", "winner_test_scatter.png",
             "winner_test_residuals.png", "test_absolute_error_cdf.png"]
    completed.write_text(json.dumps({"configuration": configuration, "output_sha256": {
        name: sha256_file(output_dir / name) for name in files}}, indent=2), encoding="utf-8")
    for name, score in metrics.items():
        print(f"{name}: test MAE {score['mae_minutes']:.4f}; RMSE {score['rmse_minutes']:.4f} minutes; R2 {score['r2']}")
    print(f"Validation-selected winner retained: {winner}; test records: {len(test):,}")
    print(f"Saved to {output_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Step 5: final frozen-model evaluation and paired source-date uncertainty")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "step02")
    parser.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "step03")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "step04")
    parser.add_argument("--prepared-input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "step05")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        evaluate_models(args.split_dir, args.features_dir, args.model_dir, args.output_dir,
                        args.prepared_input, args.bootstrap_iterations, args.seed)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Final evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
