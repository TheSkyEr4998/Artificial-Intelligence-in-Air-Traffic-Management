"""Training-range and imputation diagnostics; these are not confidence estimates."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def training_support_profile(train, factors, train_sha256):
    bounds = {}
    for factor in factors:
        if factor != "type":
            observed = pd.to_numeric(train[factor], errors="raise").dropna()
            if observed.empty or not np.isfinite(observed).all():
                raise ValueError(f"No finite training range for {factor}")
            bounds[factor] = {"minimum": float(observed.min()), "maximum": float(observed.max())}
    return {"version": 1, "train_sha256": train_sha256, "factors": list(factors), "bounds": bounds,
            "scope": "Marginal training ranges only; joint support and calibrated uncertainty are not established"}


def load_support_profile(path, selection):
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if (profile.get("version") != 1 or profile.get("train_sha256") != selection["train_sha256"]
            or profile.get("factors") != selection["selected_factors"]):
        raise ValueError("Support profile does not match the frozen training data/features")
    expected = set(profile["factors"]) - {"type"}
    if set(profile["bounds"]) != expected:
        raise ValueError("Support profile has incomplete bounds")
    for bounds in profile["bounds"].values():
        if (set(bounds) != {"minimum", "maximum"} or
                not np.isfinite(list(bounds.values())).all() or bounds["minimum"] > bounds["maximum"]):
            raise ValueError("Invalid support profile bounds")
    return profile


def input_quality(frame, factors, profile=None):
    missing = [[] for _ in range(len(frame))]
    outside = [[] for _ in range(len(frame))]
    for factor in factors:
        if factor == "type":
            continue
        values = pd.to_numeric(frame[factor], errors="raise").to_numpy(dtype=float, na_value=np.nan)
        for index in np.flatnonzero(np.isnan(values)):
            missing[index].append(factor)
        if profile:
            bounds = profile["bounds"][factor]
            mask = np.isfinite(values) & ((values < bounds["minimum"]) | (values > bounds["maximum"]))
            for index in np.flatnonzero(mask):
                outside[index].append(factor)
    return pd.DataFrame({
        "imputed_factors": [";".join(row) for row in missing],
        "outside_training_range_factors": [";".join(row) for row in outside],
        "training_range_status": [("outside_marginal_training_ranges" if row else
                                   "within_marginal_training_ranges") if profile else "not_checked"
                                  for row in outside],
    }, index=frame.index)
