"""Step 4: train on training records and select configurations on validation.

Uses the frozen Step 3 original-factor encoder for every model. Test data is not
opened. Final test evaluation is a separate stage after these choices are frozen.
"""

import argparse
import json
import math
import os
from pathlib import Path
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

from anfis import ANFISRegressor
from create_splits import TARGET, sha256_file
from preprocessing import encoded_names, fit_encoder, prediction_metrics, read_development_data, transform_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["polynomial", "neural_network", "anfis"]


def load_training_inputs(split_dir, features_dir):
    """Verify the Step 3 provenance and load only development partitions."""
    contract = json.loads((features_dir / "selected_features.json").read_text(encoding="utf-8"))
    encoder = json.loads((features_dir / "encoder.json").read_text(encoding="utf-8"))
    for file_name, key in [("train.csv", "train_sha256"), ("validation.csv", "validation_sha256"),
                           ("split_manifest.json", "split_manifest_sha256")]:
        if sha256_file(split_dir / file_name) != contract[key]:
            raise ValueError(f"{file_name} changed since Step 3; reproduce feature selection before training")
    train, validation, split_manifest = read_development_data(split_dir)
    factors = contract["selected_factors"]
    if len(factors) not in {7, 8} or encoder["factors"] != factors:
        raise ValueError("Feature-selection and encoder contracts disagree")
    if contract["encoded_columns"] != encoded_names(factors) or encoder["encoded_columns"] != encoded_names(factors):
        raise ValueError("Encoded column order disagrees with selected factors")
    # Recompute training statistics solely to detect accidental encoder edits.
    expected = fit_encoder(train, factors)
    if encoder["fit_rows"] != len(train) or encoder["hour_mode"] != expected["hour_mode"] or encoder["numeric_medians"] != expected["numeric_medians"]:
        raise ValueError("Encoder imputation values do not match these training records")
    if not np.allclose(encoder["mean"], expected["mean"], rtol=1e-10, atol=1e-10) or not np.allclose(encoder["scale"], expected["scale"], rtol=1e-10, atol=1e-10):
        raise ValueError("Encoder scaling values do not match these training records")
    x_train = transform_features(train, encoder)
    x_validation = transform_features(validation, encoder)
    return train, validation, x_train, x_validation, encoder, contract, split_manifest


def target_model(regressor):
    """Shared log1p and training-standardized target; inverse predictions in minutes."""
    transformation = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, inverse_func=np.expm1)),
        ("scale", StandardScaler()),
    ])
    return TransformedTargetRegressor(regressor=regressor, transformer=transformation)


def candidate_grid(n_features, seed=42, quick=False):
    """A small declared grid; outer validation is never used for fit stopping."""
    candidates = []
    for degree in ([2] if quick else [1, 2, 3]):
        for alpha in ([1.0] if quick else [1.0, 10.0]):
            model = Pipeline([("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
                              ("term_scaler", StandardScaler()),
                              ("ridge", Ridge(alpha=alpha, solver="cholesky"))])
            parameters = {"degree": degree, "alpha": alpha}
            candidates.append((f"polynomial_d{degree}_a{alpha:g}", "polynomial", parameters,
                               math.comb(n_features + degree, degree), target_model(model)))
    for units in ([16] if quick else [8, 16, 32]):
        parameters = {"hidden_units": units, "alpha": 0.001, "learning_rate_init": 0.001,
                      "max_epochs": 20 if quick else 500, "stopping": "training-loss tolerance only"}
        model = MLPRegressor(hidden_layer_sizes=(units,), activation="relu", solver="adam",
                             alpha=0.001, learning_rate_init=0.001, batch_size=256,
                             max_iter=parameters["max_epochs"], early_stopping=False,
                             tol=1e-4, n_iter_no_change=15, random_state=seed)
        candidates.append((f"neural_h{units}", "neural_network", parameters,
                           units * (n_features + 2) + 1, target_model(model)))
    for rules in ([4] if quick else [4, 8]):
        parameters = {"rules": rules, "epochs": 15 if quick else 25,
                      "learning_rate": 0.01, "ridge": 0.002}
        model = ANFISRegressor(n_rules=rules, epochs=parameters["epochs"],
                               learning_rate=parameters["learning_rate"], ridge=parameters["ridge"],
                               random_state=seed)
        candidates.append((f"anfis_r{rules}", "anfis", parameters,
                           rules * (3 * n_features + 1), target_model(model)))
    return candidates


def choice_key(record):
    return (record["validation_mae_minutes"], record["validation_rmse_minutes"],
            record["parameter_count"], record["candidate"])


def predict_minutes(model, values):
    """All evaluation paths share the same un-clipped, inverse-target prediction."""
    predictions = np.asarray(model.predict(values), dtype=float).reshape(-1)
    if not np.isfinite(predictions).all():
        raise ValueError("A fitted model returned nonfinite predictions")
    return predictions


def model_diagnostics(estimator, family):
    regressor = estimator.regressor_ if isinstance(estimator, TransformedTargetRegressor) else estimator
    curve = list(getattr(regressor, "loss_curve_", []))
    diagnostics = {"epochs_completed": int(getattr(regressor, "n_iter_", 0)),
                   "training_loss_curve": [float(v) for v in curve]}
    if family == "anfis":
        diagnostics.update({
            "initialization_rows": int(regressor.n_initialization_samples_),
            "fit_rows": int(regressor.n_training_samples_),
            "actual_rules": int(regressor.n_rules_), "best_training_epoch": int(regressor.best_epoch_),
            "premise_updates": int(regressor.n_premise_updates_),
            "best_training_mse": float(regressor.best_training_mse_),
            "centers": regressor.centers_.tolist(), "gaussian_widths": regressor.sigmas_.tolist(),
            "consequents": regressor.consequents_.tolist(),
        })
    return diagnostics


def make_training_plots(selected_records, all_records, diagnostics, predictions, output_dir):
    metrics = pd.DataFrame(selected_records)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, metric, title in zip(axes, ["mae_minutes", "rmse_minutes"], ["MAE", "RMSE"]):
        indices = np.arange(len(metrics))
        ax.bar(indices - .18, metrics[f"train_{metric}"], width=.36, label="Training", color="#237a91")
        ax.bar(indices + .18, metrics[f"validation_{metric}"], width=.36, label="Validation", color="#e38b35")
        ax.set_xticks(indices, metrics.family, rotation=20, ha="right")
        ax.set(ylabel=f"{title} (minutes)", title=f"Selected models: {title}")
        ax.legend()
    fig.savefig(output_dir / "training_validation_comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for record in selected_records:
        curve = diagnostics[record["candidate"]]["training_loss_curve"]
        if curve:
            epochs = np.arange(len(curve)) if record["family"] == "anfis" else np.arange(1, len(curve) + 1)
            ax.plot(epochs, curve, label=record["family"])
    ax.set(xlabel="Training epoch", ylabel="Internal standardized-log-target training loss",
           title="Training diagnostics (MLP and ANFIS losses use different conventions)")
    if ax.lines:
        ax.legend()
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)
    validation = pd.DataFrame(all_records).sort_values("validation_mae_minutes")
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    ax.bar(validation.candidate, validation.validation_mae_minutes, color="#237a91")
    ax.set(ylabel="Validation MAE (minutes)", title="Declared configuration comparison")
    ax.tick_params(axis="x", labelrotation=45)
    fig.savefig(output_dir / "validation_candidates.png", dpi=160)
    plt.close(fig)


def train_and_select(split_dir, features_dir, output_dir, seed=42, quick=False):
    # Bound this process's worker count; avoids obsolete Windows physical-core
    # probes on systems without WMIC. Existing explicit user settings are kept.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
    output_dir = Path(output_dir)
    train, validation, x_train, x_validation, encoder, contract, split_manifest = load_training_inputs(Path(split_dir), Path(features_dir))
    y_train, y_validation = train[TARGET].to_numpy(), validation[TARGET].to_numpy()
    records, diagnostics, fitted, family_best = [], {}, {}, {}
    configurations = [("median_baseline", "median_baseline", {"strategy": "median"}, 1,
                       DummyRegressor(strategy="median")), *candidate_grid(x_train.shape[1], seed, quick)]
    for candidate, family, parameters, count, model in configurations:
        print(f"Training {candidate}: {parameters}", flush=True)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with threadpool_limits(limits=2):
                model.fit(x_train, y_train)
                train_prediction = predict_minutes(model, x_train)
                validation_prediction = predict_minutes(model, x_validation)
        train_metrics = prediction_metrics(y_train, train_prediction)
        validation_metrics = prediction_metrics(y_validation, validation_prediction)
        messages = sorted({str(item.message) for item in caught})
        record = {"candidate": candidate, "family": family, "parameters": parameters,
                  "parameter_count": count, "fit_seconds": float(time.perf_counter() - started),
                  "warnings": messages,
                  **{f"train_{key}": value for key, value in train_metrics.items()},
                  **{f"validation_{key}": value for key, value in validation_metrics.items()}}
        records.append(record)
        diagnostics[candidate] = model_diagnostics(model, family)
        if family not in family_best or choice_key(record) < choice_key(family_best[family]):
            family_best[family] = record
            fitted[family] = (model, validation_prediction)
        print(f"  Validation MAE {validation_metrics['mae_minutes']:.4f}; RMSE {validation_metrics['rmse_minutes']:.4f} minutes", flush=True)
        for message in messages:
            print(f"  Training warning saved: {message}", flush=True)
    selected_records = sorted(family_best.values(), key=choice_key)
    winner = selected_records[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    model_digests = {}
    prediction_frames = []
    for record in selected_records:
        family = record["family"]
        model, predictions = fitted[family]
        artifact = {"artifact_version": 1, "model": model, "encoder": encoder,
                    "family": family, "candidate": record["candidate"], "selected_factors": contract["selected_factors"],
                    "target": TARGET, "target_units": "minutes", "prediction_clipping": "none",
                    "sklearn_version": sklearn.__version__}
        artifact_name = f"{family}.joblib"
        joblib.dump(artifact, output_dir / artifact_name)
        model_digests[family] = sha256_file(output_dir / artifact_name)
        table = validation[["id", "type", "source_date"]].copy()
        table["family"] = family
        table["actual_minutes"] = y_validation
        table["predicted_minutes"] = predictions
        table["residual_minutes"] = y_validation - predictions
        prediction_frames.append(table)
    for table_name, table in [("validation_candidates.csv", pd.DataFrame(records)),
                              ("selected_models.csv", pd.DataFrame(selected_records))]:
        table = table.copy()
        for column in ["parameters", "warnings"]:
            table[column] = table[column].map(json.dumps)
        table.to_csv(output_dir / table_name, index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    (output_dir / "training_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8")
    selection = {
        "protocol_version": 1, "seed": seed, "quick_training_budget": quick,
        "prediction_task": contract["prediction_task"], "target": TARGET, "target_units": "minutes",
        "selected_factors": contract["selected_factors"], "encoded_columns": encoder["encoded_columns"],
        "training_rows": len(train), "validation_rows": len(validation), "split_manifest": split_manifest,
        "step3_contract_sha256": sha256_file(Path(features_dir) / "selected_features.json"),
        "encoder_sha256": sha256_file(Path(features_dir) / "encoder.json"),
        "train_sha256": contract["train_sha256"], "validation_sha256": contract["validation_sha256"],
        "selection_rule": "validation MAE first, RMSE second, parameter count then name for exact ties",
        "winner_family": winner["family"], "winner_candidate": winner["candidate"],
        "selected_models": selected_records, "fitted_model_sha256": model_digests,
        "source_sha256": {name: sha256_file(Path(__file__).resolve().parent / name)
                          for name in ["train_models.py", "anfis.py", "preprocessing.py"]},
        "target_transform": "log1p then training-fitted standard scaler for three learned families; inverse to minutes",
        "prediction_clipping": "none", "fit_on_validation": False,
        "test_status": "not opened, transformed, scored or hashed by Step 4",
        "notes": [
            "Every learned model uses all outer-training records and the same frozen original-factor encoder",
            "MLP stopping uses training loss only; no random internal validation or outer-validation epoch selection",
            "ANFIS initial clustering can use a deterministic training subsample; hybrid fitting uses all training rows",
            "Selection scores are development evidence: validation already helped choose the features",
            "Small differences are not established statistical superiority or globally optimal settings",
            "Heavy unverified positive source labels remain; no duration cutoff was tuned against model scores",
            "Saved models stay trained on the original training block; validation is not added after selection",
            "This completed-journey benchmark does not establish live taxi-start forecasting accuracy",
        ],
    }
    (output_dir / "model_selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False), encoding="utf-8")
    make_training_plots(selected_records, records, diagnostics, predictions, output_dir)
    print(f"Selected on validation: {winner['family']} ({winner['candidate']})", flush=True)
    print(f"Validation MAE: {winner['validation_mae_minutes']:.4f}; RMSE: {winner['validation_rmse_minutes']:.4f} minutes", flush=True)
    print("Model choices saved. Test data remains reserved for Step 5.", flush=True)
    print(f"Saved to {output_dir}", flush=True)
    return selection


def main():
    parser = argparse.ArgumentParser(description="Step 4: fit models and freeze validation selections without opening test data")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "step02")
    parser.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "step03")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "step04")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Reduced-budget smoke run; not the complete configuration comparison")
    args = parser.parse_args()
    try:
        train_and_select(args.split_dir, args.features_dir, args.output_dir, args.seed, args.quick)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Model training failed: {error}\n")


if __name__ == "__main__":
    main()
