"""Predict minutes for completed-journey feature records using a frozen model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from create_splits import sha256_file
from model_io import load_frozen_models, predict_from_artifact
from prediction_quality import input_quality, load_support_profile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_FACTORS = {
    "start_hour": ("gate (block) hour", "runway hour"),
    "n_departing_at_start": ("NDepDep", "NArrDep"),
    "n_arriving_at_start": ("NDepArr", "NArrArr"),
}


def normalize_start_factors(frame):
    """Optionally derive the same start-side factors from the supplied raw schema."""
    result = frame.copy()
    if "type" not in result:
        raise ValueError("Required predictor column missing: type")
    result["type"] = result["type"].astype("string").str.strip().str.lower()
    departure = result["type"].eq("departure")
    for derived, (departure_field, arrival_field) in NORMALIZED_FACTORS.items():
        if derived not in result and departure_field in result and arrival_field in result:
            result[derived] = np.where(departure, result[departure_field], result[arrival_field])
    return result


def predict_file(input_path, output_path, model_dir, family=None, support_profile=None):
    input_path, output_path, model_dir = map(Path, [input_path, output_path, model_dir])
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Prediction output must differ from the input file")
    selection, artifacts = load_frozen_models(model_dir)
    chosen = family or selection["winner_family"]
    if chosen not in artifacts:
        raise ValueError(f"Unknown model family: {chosen}")
    frame = pd.read_csv(input_path)
    normalized = normalize_start_factors(frame)
    with threadpool_limits(limits=2):
        predictions = predict_from_artifact(normalized, artifacts[chosen])
    profile_path = Path(support_profile) if support_profile else model_dir.parent / "step06" / "training_support.json"
    if support_profile and not profile_path.is_file():
        raise ValueError(f"Support profile missing: {profile_path}")
    profile = load_support_profile(profile_path, selection) if profile_path.is_file() else None
    diagnostics = input_quality(normalized, artifacts[chosen]["selected_factors"], profile)
    output = frame.copy()
    output["predicted_taxi_time_minutes"] = predictions
    output["model_family"] = chosen
    output["prediction_status"] = np.where(predictions < 0, "invalid_negative_duration", "finite_prediction")
    for column in diagnostics:
        output[column] = diagnostics[column]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    (output_path.parent / f"{output_path.stem}_metadata.json").write_text(json.dumps({
        "family": chosen, "candidate": artifacts[chosen]["candidate"], "rows": len(output),
        "input_sha256": sha256_file(input_path), "output_sha256": sha256_file(output_path),
        "model_sha256": selection["fitted_model_sha256"][chosen],
        "negative_prediction_count": int(np.sum(predictions < 0)),
        "imputed_record_count": int(diagnostics.imputed_factors.ne("").sum()),
        "training_range_checked": profile is not None,
        "training_range_note": "Marginal ranges only; they do not establish joint support or calibrated confidence",
        "outside_training_range_count": int(diagnostics.outside_training_range_factors.ne("").sum()) if profile else None,
        "support_profile_sha256": sha256_file(profile_path) if profile else None,
        "target_used_as_predictor": False, "target_units": "minutes", "prediction_clipping": "none",
        "scope": "retrospective completed-journey feature prediction; no live taxi-start accuracy established",
        "note": "A finite prediction status is numerical validity, not a calibrated confidence or accuracy claim",
    }, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Predicted {len(output):,} records with {chosen}; saved to {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Apply this project's frozen taxi-time model to feature records")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "predictions" / "predictions.csv")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "step04")
    parser.add_argument("--family", choices=["polynomial", "neural_network", "anfis", "median_baseline"])
    parser.add_argument("--support-profile", type=Path, help="Optional training support JSON produced by quality_audit.py")
    args = parser.parse_args()
    try:
        predict_file(args.input, args.output, args.model_dir, args.family, args.support_profile)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Prediction failed: {error}\n")


if __name__ == "__main__":
    main()
