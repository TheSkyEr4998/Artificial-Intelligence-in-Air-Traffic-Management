# Start here

This repository contains the complete Python workflow. Its modelling input is the 12 monthly `features_*.csv` files. Cleaned observations, snapped CSVs, the airport map, and supplied executables are used only for source auditing.

1. Keep the data folders outside Git and install the dependencies with the commands in [README.md](README.md).
2. Run the test suite.
3. Run `scripts/run_full_project.ps1` once with the four data-folder paths and a new run name.
4. Inspect `artifacts/runs/<run-name>/run_summary.json`, source-review outputs, model metrics, prediction example, and `publication_figures/<run-name>/export_manifest.csv`.

The run refuses to overwrite an existing run directory. Keep the original project files and use a new run name if you need to repeat it. The December evaluation dates were previously inspected, so the generated result is labelled repeated development evaluation.
