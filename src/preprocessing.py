"""Step 3: training-only PCA diagnostics and a limited validation comparison.

Four context factors are retained by design; PCA guides three/four route factors.
Test records are deliberately not loaded. This is feature selection for a fixed
probe, not the final neural-network / polynomial / ANFIS model comparison.
"""

import argparse
from itertools import combinations
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

from create_splits import FEATURES, TARGET, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ["type", "start_hour", "n_departing_at_start", "n_arriving_at_start"]
ROUTE = ["distance", "angle", "shortest path", "distance_gate", "distance_long", "distance_else"]
DISTANCE_IDENTITY = {"distance", "distance_gate", "distance_long", "distance_else"}


def read_development_data(split_dir: Path):
    """Read train + validation and the split contract; never open test.csv."""
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest["candidate_features"] != FEATURES:
        raise ValueError("Unexpected Step 2 feature contract")
    frames = {}
    for name in ["train", "validation"]:
        frame = pd.read_csv(split_dir / f"{name}.csv", dtype={"id": "string", "type": "string"})
        required = ["id", "source_date", TARGET, *FEATURES]
        if not set(required).issubset(frame.columns):
            raise ValueError(f"Required columns missing from {name}.csv")
        frame["source_date"] = pd.to_datetime(frame["source_date"], format="ISO8601", errors="coerce")
        if frame.empty or frame["source_date"].isna().any() or frame["id"].isna().any():
            raise ValueError(f"Empty partition or invalid metadata in {name}.csv")
        if frame.duplicated(["id", "type"]).any() or not frame["type"].isin(["arrival", "departure"]).all():
            raise ValueError(f"Invalid keys or movement types in {name}.csv")
        for feature in [TARGET, *[f for f in FEATURES if f != "type"]]:
            frame[feature] = pd.to_numeric(frame[feature], errors="raise")
        if not (np.isfinite(frame[TARGET]) & frame[TARGET].gt(0)).all():
            raise ValueError(f"Invalid target in {name}.csv")
        details = manifest["partitions"][name]
        if len(frame) != details["rows"] or frame.source_date.min().strftime("%Y-%m-%d") != details["first_date"] or frame.source_date.max().strftime("%Y-%m-%d") != details["last_date"]:
            raise ValueError(f"{name}.csv does not match the Step 2 manifest")
        frames[name] = frame
    if frames["train"].source_date.max() >= frames["validation"].source_date.min():
        raise ValueError("Training and validation dates overlap")
    if not frames["train"].merge(frames["validation"], on=["id", "type"]).empty:
        raise ValueError("Training and validation share record keys")
    return frames["train"], frames["validation"], manifest


def encoded_names(factors):
    names = []
    for factor in factors:
        names.extend(["start_hour_sin", "start_hour_cos"] if factor == "start_hour" else
                     ["is_departure"] if factor == "type" else [factor])
    return names


def encode_values(frame, state):
    """Use recorded training imputation values and deterministic domain encoding."""
    columns = []
    for factor in state["factors"]:
        if factor not in frame:
            raise ValueError(f"Predictor missing: {factor}")
        if factor == "type":
            if not frame[factor].isin(["arrival", "departure"]).all():
                raise ValueError("type must be arrival or departure")
            columns.append(frame[factor].eq("departure").to_numpy(dtype=float))
        else:
            values = pd.to_numeric(frame[factor], errors="raise").to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError(f"Nonfinite predictor {factor}; apply Step 2 validation first")
            fill = state["hour_mode"] if factor == "start_hour" else state["numeric_medians"][factor]
            values = np.where(np.isnan(values), fill, values)
            if factor == "start_hour":
                if ((values < 0) | (values > 23) | (values % 1 != 0)).any():
                    raise ValueError("start_hour must be an integer from 0 to 23")
                radians = values * (2 * np.pi / 24)
                columns.extend([np.sin(radians), np.cos(radians)])
            else:
                columns.append(values)
    return np.column_stack(columns)


def fit_encoder(train, factors):
    """Learn medians, hour mode, means and scales from training records only."""
    if not factors or len(set(factors)) != len(factors) or not set(factors).issubset(FEATURES):
        raise ValueError("Factors must be unique members of the Step 2 predictor allowlist")
    state = {"factors": list(factors), "encoded_columns": encoded_names(factors), "numeric_medians": {}}
    hour = pd.to_numeric(train["start_hour"], errors="raise").dropna()
    if hour.empty:
        raise ValueError("No observed training start hours")
    state["hour_mode"] = float(hour.mode().iloc[0])
    for factor in factors:
        if factor not in {"type", "start_hour"}:
            values = pd.to_numeric(train[factor], errors="raise")
            median = values.median()
            if not np.isfinite(median):
                raise ValueError(f"No finite training median for {factor}")
            state["numeric_medians"][factor] = float(median)
    scaler = StandardScaler().fit(encode_values(train, state))
    state["mean"] = scaler.mean_.tolist()
    state["scale"] = scaler.scale_.tolist()
    state["fit_rows"] = len(train)
    return state


def transform_features(frame, state):
    values = (encode_values(frame, state) - np.asarray(state["mean"])) / np.asarray(state["scale"])
    if not np.isfinite(values).all():
        raise ValueError("Encoded predictors are not finite")
    return values


def fit_route_pca(train, variance_threshold=0.95):
    """Inspect standardized route variables, separately from fixed context factors."""
    if not 0 < variance_threshold <= 1:
        raise ValueError("PCA variance threshold must be in (0, 1]")
    if len(train) <= len(ROUTE) or train[ROUTE].isna().all().any():
        raise ValueError("Insufficient training route observations for PCA")
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()), ("pca", PCA(svd_solver="full"))])
    pipeline.fit(train[ROUTE])
    z = pipeline["scaler"].transform(pipeline["imputer"].transform(train[ROUTE]))
    if np.any(np.var(z, axis=0) == 0):
        raise ValueError("A route feature is constant in training; review the candidate pool")
    pca = pipeline["pca"]
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    retained = min(len(ROUTE), int(np.searchsorted(cumulative, variance_threshold) + 1))
    scores = pca.transform(z)[:, :retained]
    variance = pd.DataFrame({"component": [f"PC{i+1}" for i in range(len(ROUTE))],
                             "variance_ratio": pca.explained_variance_ratio_,
                             "cumulative_variance": cumulative,
                             "retained": np.arange(len(ROUTE)) < retained})
    # Variable variance represented by retained PCs: descriptive, not target importance.
    represented = ((pca.components_[:retained].T ** 2) @
                   (pca.explained_variance_[:retained] * (len(train) - 1) / len(train)))
    coverage = pd.DataFrame({"factor": ROUTE, "variance_represented_by_retained_pcs": represented})
    return pipeline, z, scores, retained, variance, coverage


def shortlist_routes(z, retained_scores, retained_fraction, per_size=3, distance_identity_verified=False):
    """Rank original-variable subsets by training reconstruction of retained PCs.

    All eligible subsets are ranked without taxi_time or validation records.
    Only a small predeclared shortlist per size proceeds to predictive comparison.
    """
    if not 1 <= per_size <= 5:
        raise ValueError("shortlist-per-size must be from 1 to 5")
    denominator = float(np.sum(retained_scores ** 2))
    if denominator <= 0:
        raise ValueError("No retained route variation")
    rows = []
    representatives = {}
    for size in [3, 4]:
        for chosen in combinations(ROUTE, size):
            if DISTANCE_IDENTITY.issubset(chosen):
                continue
            selected = z[:, [ROUTE.index(f) for f in chosen]]
            design = np.column_stack([np.ones(len(z)), selected])
            coefficients = np.linalg.lstsq(design, retained_scores, rcond=None)[0]
            loss = float(np.sum((retained_scores - design @ coefficients) ** 2) / denominator)
            signature = tuple(chosen)
            if distance_identity_verified and len(set(chosen) & DISTANCE_IDENTITY) == 3:
                # Any three of these four columns span the same distance space.
                # Retain the first combination in the declared ROUTE order.
                signature = ("distance_basis_rank3", *[f for f in chosen if f not in DISTANCE_IDENTITY])
            equivalence_key = (size, signature)
            equivalent_to = representatives.get(equivalence_key, "")
            representatives.setdefault(equivalence_key, json.dumps(list(chosen)))
            rows.append({"route_factors": list(chosen), "factor_count": len(CONTEXT) + size,
                         "retained_pc_reconstruction_loss": loss,
                         "total_route_variance_loss": 1 - retained_fraction * (1 - loss),
                         "equivalent_to": equivalent_to})
    table = pd.DataFrame(rows)
    table["tie_key"] = table.route_factors.map(lambda items: "|".join(str(ROUTE.index(f)) for f in items))
    table["rounded_loss"] = table.retained_pc_reconstruction_loss.round(12)
    table = table.sort_values(["factor_count", "rounded_loss", "tie_key"], kind="stable").reset_index(drop=True)
    unique = table.equivalent_to.eq("")
    table["pca_rank_within_size"] = 0
    table.loc[unique, "pca_rank_within_size"] = table.loc[unique].groupby("factor_count").cumcount() + 1
    table["shortlisted"] = unique & table.pca_rank_within_size.le(per_size)
    return table.drop(columns=["rounded_loss", "tie_key"])


def make_probe():
    """Fixed quadratic Ridge with a predeclared log1p target transformation."""
    regression = Pipeline([("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                           ("polynomial_scaler", StandardScaler()),
                           ("ridge", Ridge(alpha=1.0, solver="cholesky"))])
    return TransformedTargetRegressor(regressor=regression, func=np.log1p, inverse_func=np.expm1)


def prediction_metrics(actual, predicted):
    if not np.isfinite(predicted).all():
        raise ValueError("Probe produced nonfinite predictions; no scores were silently dropped")
    return {"mae_minutes": float(mean_absolute_error(actual, predicted)),
            "rmse_minutes": float(root_mean_squared_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted)),
            "negative_prediction_count": int(np.sum(predicted < 0))}


def compare_subsets(train, validation, shortlist):
    rows, fitted = [], {}
    baseline_value = float(train[TARGET].median())
    baseline = {"training_median_minutes": baseline_value,
                **prediction_metrics(validation[TARGET], np.full(len(validation), baseline_value))}
    for item in shortlist.loc[shortlist.shortlisted].itertuples(index=False):
        factors = [*CONTEXT, *item.route_factors]
        name = f"{len(factors)}f_pca{item.pca_rank_within_size}"
        encoder = fit_encoder(train, factors)
        model = make_probe()
        model.fit(transform_features(train, encoder), train[TARGET])
        predictions = model.predict(transform_features(validation, encoder))
        metrics = prediction_metrics(validation[TARGET], predictions)
        rows.append({"candidate": name, "factor_count": len(factors), "factors": factors,
                     "encoded_column_count": len(encoder["encoded_columns"]),
                     "pca_rank_within_size": item.pca_rank_within_size,
                     "total_route_variance_loss": item.total_route_variance_loss, **metrics})
        fitted[name] = (encoder, model)
    results = pd.DataFrame(rows).sort_values(["mae_minutes", "rmse_minutes", "factor_count", "candidate"], kind="stable").reset_index(drop=True)
    selected = results.iloc[0].to_dict()
    return results, selected, fitted[selected["candidate"]], baseline


def make_plots(variance, pca, retained, results, baseline, training_scores, output_dir):
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.bar(variance.component, variance.variance_ratio * 100, color="#237a91", label="Individual")
    ax.plot(variance.component, variance.cumulative_variance * 100, "o-", color="#e38b35", label="Cumulative")
    ax.set(ylabel="Training route variance (%)", xlabel="Principal component", ylim=(0, 105),
           title=f"Route PCA: {retained} components retained")
    ax.legend()
    fig.savefig(output_dir / "pca_variance.png", dpi=160)
    plt.close(fig)
    weights = pca.components_[:retained]
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    view = ax.imshow(weights, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(ROUTE)), ROUTE, rotation=25, ha="right")
    ax.set_yticks(np.arange(retained), [f"PC{i+1}" for i in range(retained)])
    ax.set_title("Training PCA weights for standardized route variables")
    for i in range(retained):
        for j in range(len(ROUTE)):
            ax.text(j, i, f"{weights[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(view, ax=ax, label="PCA coefficient")
    fig.savefig(output_dir / "pca_weights.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    if training_scores.shape[1] >= 2:
        view = ax.hexbin(training_scores[:, 0], training_scores[:, 1], gridsize=45, bins="log", mincnt=1, cmap="viridis")
        fig.colorbar(view, ax=ax, label="Training records per bin")
        ax.set(xlabel="PC1 score", ylabel="PC2 score")
    else:
        ax.hist(training_scores[:, 0], bins=45, color="#237a91")
        ax.set(xlabel="PC1 score", ylabel="Training records")
    ax.set_title("Training route PCA projection (diagnostic)")
    fig.savefig(output_dir / "training_pca_projection.png", dpi=160)
    plt.close(fig)
    ordered = results.sort_values(["factor_count", "pca_rank_within_size"])
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.bar(ordered.candidate, ordered.mae_minutes, color=["#237a91" if n == 7 else "#e38b35" for n in ordered.factor_count])
    ax.axhline(baseline["mae_minutes"], color="#333333", linestyle="--", label="Training-median baseline")
    ax.set(ylabel="Validation MAE (minutes)", xlabel="PCA-shortlisted feature choice",
           title="Fixed quadratic Ridge probe with log1p target")
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend()
    fig.savefig(output_dir / "validation_feature_comparison.png", dpi=160)
    plt.close(fig)


def save_table_with_lists(table, path):
    table = table.copy()
    for column in ["factors", "route_factors"]:
        if column in table:
            table[column] = table[column].map(json.dumps)
    table.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Step 3: PCA-guided original factor selection without opening test data")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "step02")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "step03")
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--shortlist-per-size", type=int, default=3)
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=2):
            train, validation, split_manifest = read_development_data(args.split_dir)
            pca_pipeline, z, scores, retained, variance, coverage = fit_route_pca(train, args.pca_variance)
            retained_fraction = float(variance.variance_ratio.iloc[:retained].sum())
            identity_verified = bool(train[list(DISTANCE_IDENTITY)].notna().all().all() and
                                     np.allclose(train["distance"], train[["distance_gate", "distance_long", "distance_else"]].sum(axis=1), rtol=1e-9, atol=1e-8))
            shortlist = shortlist_routes(z, scores, retained_fraction, args.shortlist_per_size, identity_verified)
            results, selected, (encoder, probe), baseline = compare_subsets(train, validation, shortlist)
        selected = {key: value.item() if isinstance(value, np.generic) else value for key, value in selected.items()}
        contract = {
            "protocol_version": 1, "prediction_task": split_manifest["prediction_task"],
            "target": TARGET, "target_units": "minutes", "selected_factors": selected["factors"],
            "factor_count": selected["factor_count"], "encoded_columns": encoder["encoded_columns"],
            "context_factors_fixed_by_design": CONTEXT,
            "route_candidates": ROUTE, "pca_retained_components": retained,
            "pca_retained_route_variance": retained_fraction,
            "pca_threshold": args.pca_variance, "shortlist_per_size": args.shortlist_per_size,
            "distance_identity_verified_on_training": identity_verified,
            "equivalent_route_subsets_collapsed": int(shortlist.equivalent_to.ne("").sum()),
            "train_sha256": sha256_file(args.split_dir / "train.csv"),
            "validation_sha256": sha256_file(args.split_dir / "validation.csv"),
            "split_manifest_sha256": sha256_file(args.split_dir / "split_manifest.json"),
            "selection_rule": "validation MAE first; RMSE then fewer factors for exact ties",
            "probe": {"model": "quadratic Ridge", "degree": 2, "alpha": 1.0,
                      "target_transform": "log1p; inverse expm1; scores in minutes", "prediction_clipping": "none"},
            "validation_result": selected, "median_baseline": baseline,
            "beats_median_on_mae": bool(selected["mae_minutes"] < baseline["mae_minutes"]),
            "test_status": "not opened, transformed, scored or hashed by Step 3",
            "notes": [
                "Hybrid domain and PCA-guided selection of original factors; PCA does not select all context factors",
                "PCA scores are diagnostic; final models use selected original factors with recorded encoding",
                "Equivalent distance bases are collapsed before shortlisting using the first declared route-order representative",
                "All imputers, scalers, PCA, reconstruction coefficients and probes are fitted on training only",
                "Selection is conditional on this fixed probe and validation period; no global best-subset claim",
                "Positive duration extremes are retained; inferred source labels still need quality investigation",
                "Completed route predictors support retrospective modelling, not verified taxi-start forecasting",
            ],
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        variance.to_csv(args.output_dir / "route_pca_variance.csv", index=False)
        coverage.to_csv(args.output_dir / "route_factor_variance_coverage.csv", index=False)
        score_table = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(retained)])
        score_table.insert(0, "source_date", train.source_date.to_numpy())
        score_table.insert(0, "id", train.id.to_numpy())
        score_table.to_csv(args.output_dir / "training_route_pca_scores.csv", index=False)
        pd.DataFrame(pca_pipeline["pca"].components_, columns=ROUTE,
                     index=[f"PC{i+1}" for i in range(len(ROUTE))]).rename_axis("component").to_csv(args.output_dir / "route_pca_weights.csv")
        save_table_with_lists(shortlist, args.output_dir / "route_subset_reconstruction.csv")
        save_table_with_lists(results, args.output_dir / "validation_feature_comparison.csv")
        for name, frame in [("train", train), ("validation", validation)]:
            frame[["id", "source_date", TARGET, *selected["factors"]]].to_csv(args.output_dir / f"{name}_selected.csv", index=False)
            encoded = pd.DataFrame(transform_features(frame, encoder), columns=encoder["encoded_columns"])
            encoded.insert(0, TARGET, frame[TARGET].to_numpy())
            encoded.to_csv(args.output_dir / f"{name}_encoded.csv", index=False)
        (args.output_dir / "encoder.json").write_text(json.dumps(encoder, indent=2, allow_nan=False), encoding="utf-8")
        (args.output_dir / "selected_features.json").write_text(json.dumps(contract, indent=2, allow_nan=False), encoding="utf-8")
        joblib.dump(pca_pipeline, args.output_dir / "route_pca.joblib")
        joblib.dump(probe, args.output_dir / "validation_probe.joblib")
        make_plots(variance, pca_pipeline["pca"], retained, results, baseline, scores, args.output_dir)
        print(f"Training-only route PCA: {retained} components retain {retained_fraction:.2%} variance")
        print(f"Validation candidates compared: {len(results)}")
        print(f"Selected by the fixed validation probe: {selected['factor_count']} factors")
        for factor in selected["factors"]:
            print(f"  {factor}")
        print(f"Validation MAE: {selected['mae_minutes']:.4f} minutes; RMSE: {selected['rmse_minutes']:.4f} minutes")
        print(f"Training-median baseline validation MAE: {baseline['mae_minutes']:.4f} minutes")
        if not contract["beats_median_on_mae"]:
            print("Probe did not beat the median baseline; investigate labels and model fit before claiming useful prediction")
        print("Test data remains reserved; final comparison is performed in the evaluation stage")
        print(f"Saved to {args.output_dir}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Preprocessing failed: {error}\n")


if __name__ == "__main__":
    main()
