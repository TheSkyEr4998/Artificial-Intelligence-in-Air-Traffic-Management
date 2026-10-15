"""Public, clearly synthetic end-to-end demonstration; no real-flight accuracy claim."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from data_preparation import MONTHS, RAW_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_demo_data(directory, seed=42):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Synthetic data directory must be empty; use a new --output-dir")
    rng = np.random.default_rng(seed)
    for month_name, month in MONTHS.items():
        rows = []
        for day in [2, 8, 14, 20, 26]:
            date = pd.Timestamp(year=2021, month=month, day=day)
            for flight in range(20):
                departure = flight % 2 == 0
                gate, long, elsewhere = rng.uniform(10, 200), rng.uniform(0, 1300), rng.uniform(300, 2200)
                hour, ndep, narr = int(rng.integers(0, 24)), int(rng.poisson(3)), int(rng.poisson(2))
                duration = max(.5, 3 + 6*departure + .002*(gate+long+elsewhere)
                               + .5*ndep + .2*narr + rng.normal(0, .8))
                row = {column: 0 for column in RAW_COLUMNS}
                row.update({"id": f"{date:%d-%b-%Y}_DEMO{flight:03}", "type": "departure" if departure else "arrival",
                            "taxi_time": duration, "gate (block) hour": hour, "runway hour": hour,
                            "rwy_day": day, "rwy": "23R/05L", "distance": gate+long+elsewhere,
                            "distance_gate": gate, "distance_long": long, "distance_else": elsewhere,
                            "shortest path": (gate+long+elsewhere)*rng.uniform(.6, .95),
                            "angle": rng.uniform(100, 1000), "NDepDep": ndep, "NArrDep": ndep,
                            "NDepArr": narr, "NArrArr": narr})
                rows.append(row)
        pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(directory / f"features_{month_name}.csv", index=False)


def run_demo(output_dir):
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Demo output must be empty; choose a new --output-dir")
    generate_demo_data(root / "Feature data")
    (root / "demo_context.json").write_text(json.dumps({"data": "synthetic generated data",
        "real_flight_accuracy_established": False, "model_grid": "reduced smoke-run budget",
        "note": "This exercises software interfaces only; do not quote these scores as project performance"}, indent=2), encoding="utf-8")
    source = Path(__file__).resolve().parent
    commands = [
        ["data_preparation.py", "--data-dir", root / "Feature data", "--output-dir", root / "step01"],
        ["create_splits.py", "--input", root / "step01/features_prepared.csv", "--output-dir", root / "step02"],
        ["preprocessing.py", "--split-dir", root / "step02", "--output-dir", root / "step03"],
        ["train_models.py", "--split-dir", root / "step02", "--features-dir", root / "step03", "--output-dir", root / "step04", "--quick"],
        ["evaluate.py", "--split-dir", root / "step02", "--features-dir", root / "step03", "--model-dir", root / "step04", "--output-dir", root / "step05"],
        ["predict.py", "--model-dir", root / "step04", "--input", PROJECT_ROOT / "examples/prediction_input.csv", "--output", root / "predictions/example_predictions.csv"],
    ]
    print("SYNTHETIC SOFTWARE DEMO: these scores are not real-flight performance.", flush=True)
    for script, *args in commands:
        subprocess.run([sys.executable, str(source / script), *map(str, args)], check=True)
    print(f"Synthetic demonstration complete: {root}")


def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline on generated synthetic records without private data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "demo")
    args = parser.parse_args()
    try:
        run_demo(args.output_dir)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Synthetic demo failed: {error}\n")


if __name__ == "__main__":
    main()
