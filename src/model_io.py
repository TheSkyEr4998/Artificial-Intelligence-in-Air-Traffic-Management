"""Load this project's frozen, trusted model artifacts and validate inputs."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from create_splits import COUNTS, DISTANCES, TARGET, sha256_file
from preprocessing import encoded_names, transform_features
from train_models import choice_key, predict_minutes


KNOWN_FAMILIES = {"polynomial", "neural_network", "anfis", "median_baseline"}


def load_frozen_models(model_dir, source_dir=None):
    """Check provenance before loading artifacts produced by the trusted Step 4 run.

    Digests detect changes relative to that manifest; they do not make arbitrary
    third-party joblib files safe. This loader accepts our known artifact schema.
    """
    model_dir = Path(model_dir)
    source_dir = Path(source_dir) if source_dir else Path(__file__).resolve().parent
    selection = json.loads((model_dir / "model_selection.json").read_text(encoding="utf-8"))
    if selection["target"] != TARGET or selection["target_units"] != "minutes" or selection["fit_on_validation"]:
        raise ValueError("Unexpected frozen training protocol")
    records = selection["selected_models"]
    families = [item["family"] for item in records]
    if set(families) != KNOWN_FAMILIES or len(families) != len(KNOWN_FAMILIES):
        raise ValueError("Expected one selected model per declared family and baseline")
    winner = min(records, key=choice_key)
    if winner["family"] != selection["winner_family"] or winner["candidate"] != selection["winner_candidate"]:
        raise ValueError("Winner does not match the saved validation selection rule")
    # Check all bytes before any joblib unpickling or test-label access.
    if set(selection["source_sha256"]) != {"train_models.py", "anfis.py", "preprocessing.py"} or set(selection["fitted_model_sha256"]) != KNOWN_FAMILIES:
        raise ValueError("Incomplete frozen source/model fingerprints")
    for name, digest in selection["source_sha256"].items():
        if name not in {"train_models.py", "anfis.py", "preprocessing.py"} or sha256_file(source_dir / name) != digest:
            raise ValueError(f"Training source changed: {name}; preserve the code used for the frozen models")
    for family in families:
        if sha256_file(model_dir / f"{family}.joblib") != selection["fitted_model_sha256"][family]:
            raise ValueError(f"Frozen model changed: {family}")
    artifacts = {}
    for item in records:
        family = item["family"]
        artifact = joblib.load(model_dir / f"{family}.joblib")
        if artifact.get("artifact_version") != 1 or artifact.get("family") != family or artifact.get("candidate") != item["candidate"]:
            raise ValueError(f"Unexpected model artifact: {family}")
        if artifact.get("target") != TARGET or artifact.get("target_units") != "minutes" or artifact.get("prediction_clipping") != "none":
            raise ValueError(f"Prediction contract changed: {family}")
        encoder = artifact["encoder"]
        if artifact["selected_factors"] != selection["selected_factors"] or encoder["factors"] != selection["selected_factors"] or encoder["encoded_columns"] != selection["encoded_columns"] or encoder["encoded_columns"] != encoded_names(encoder["factors"]):
            raise ValueError(f"Feature contract changed: {family}")
        if encoder["fit_rows"] != selection["training_rows"]:
            raise ValueError(f"Encoder training population changed: {family}")
        if artifacts and encoder != next(iter(artifacts.values()))["encoder"]:
            raise ValueError("Model families do not share the same frozen encoder")
        artifacts[family] = artifact
    return selection, artifacts


def validate_prediction_inputs(frame, factors):
    """Require factor names; missing numeric cells use frozen training imputation."""
    missing = set(factors).difference(frame.columns)
    if missing:
        raise ValueError(f"Required predictor columns missing: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Prediction input has no records")
    result = frame.copy()
    result["type"] = result["type"].astype("string").str.strip().str.lower()
    if not result["type"].isin(["arrival", "departure"]).all():
        raise ValueError("type must be arrival or departure for every record")
    for factor in factors:
        if factor == "type":
            continue
        values = pd.to_numeric(result[factor], errors="raise").to_numpy(dtype=float, na_value=np.nan)
        result[factor] = values
        if np.isinf(values).any():
            raise ValueError(f"Infinite predictor: {factor}")
        observed = values[~np.isnan(values)]
        if factor in DISTANCES and (observed < 0).any():
            raise ValueError(f"{factor} cannot be negative")
        if factor in COUNTS and ((observed < 0) | (observed % 1 != 0)).any():
            raise ValueError(f"{factor} must contain nonnegative integer counts")
        if factor == "start_hour" and ((observed < 0) | (observed > 23) | (observed % 1 != 0)).any():
            raise ValueError("start_hour must contain integers from 0 to 23")
    return result


def predict_from_artifact(frame, artifact):
    checked = validate_prediction_inputs(frame, artifact["selected_factors"])
    return predict_minutes(artifact["model"], transform_features(checked, artifact["encoder"]))
