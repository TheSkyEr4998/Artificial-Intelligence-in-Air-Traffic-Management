"""Export captioned copies of figures from a completed run.

Source plots remain unchanged. Each public copy carries its cohort, evaluation,
and label-scope context, with a manifest linking source and export digests.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FIGURES = {
    "step03": ("pca_variance.png", "pca_weights.png", "training_pca_projection.png", "validation_feature_comparison.png"),
    "step04": ("validation_candidates.png", "training_validation_comparison.png", "learning_curves.png"),
    "step05": ("test_model_comparison.png", "winner_test_scatter.png", "winner_test_residuals.png", "test_absolute_error_cdf.png"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size):
    for candidate in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _captioned_png(source, captions):
    image = Image.open(source).convert("RGB")
    width, height = image.size
    font = _font(max(14, width // 88))
    padding = max(14, width // 80)
    line_height = max(22, width // 48)
    footer_height = padding * 2 + line_height * len(captions)
    output = Image.new("RGB", (width, height + footer_height), "white")
    output.paste(image, (0, 0))
    draw = ImageDraw.Draw(output)
    draw.line((0, height, width, height), fill="#cccccc", width=1)
    for index, caption in enumerate(captions):
        draw.text((padding, height + padding + index * line_height), caption, fill="#333333", font=font)
    stream = io.BytesIO()
    output.save(stream, format="PNG", optimize=False)
    return stream.getvalue(), image.size, output.size


def export_run_figures(run_dir, output_dir):
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if summary.get("test_is_fresh_independent_evidence") is not False:
        raise ValueError("Run summary must disclose that evaluation dates were previously inspected")
    validation = json.loads((run_dir / "source_validation/session_validation_summary.json").read_text(encoding="utf-8-sig"))
    cohort = json.loads((run_dir / "cohort/cohort_summary.json").read_text(encoding="utf-8-sig"))
    splits = json.loads((run_dir / "step02/split_manifest.json").read_text(encoding="utf-8-sig"))
    output_dir.mkdir(parents=True, exist_ok=True)
    records, payloads = [], []
    for stage, names in FIGURES.items():
        for name in names:
            source = run_dir / stage / name
            if not source.is_file():
                raise ValueError(f"Missing run figure: {source}")
            repeated = stage == "step05"
            counts = splits["partitions"]
            lines = [
                f"Source-session cohort: {cohort['supported_rows']:,} retained; {cohort['excluded_for_source_review_rows']:,} held for review from {cohort['input_rows']:,} prepared records.",
                (f"Repeated development evaluation; dates previously inspected. {counts['test']['rows']:,} records across {counts['test']['date_groups']} source dates; validation-selected {summary['winner_candidate']}.") if repeated else f"Training: {counts['train']['rows']:,}; validation: {counts['validation']['rows']:,}; PCA fit on training data only.",
                "Retrospective completed-route modelling; labels are inferred event durations, not official ground truth.",
            ]
            encoded, source_size, export_size = _captioned_png(source, lines)
            relative = f"{stage}_{name}"
            row = {"stage": stage, "source_relative_path": f"{stage}/{name}",
                   "export_relative_path": relative, "source_sha256": sha256(source),
                   "export_sha256": hashlib.sha256(encoded).hexdigest(),
                   "source_width": source_size[0], "source_height": source_size[1],
                   "export_width": export_size[0], "export_height": export_size[1],
                   "qualifier": "Repeated development evaluation; dates previously inspected" if repeated else "Training/validation development",
                   "scope": "Retrospective completed-route modelling; labels are inferred event durations.",
                   "caption_lines": lines}
            records.append(row)
            payloads.append((relative, encoded))
    fields = ["stage", "source_relative_path", "export_relative_path", "source_sha256", "export_sha256",
              "source_width", "source_height", "export_width", "export_height", "qualifier", "scope"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    csv_bytes = stream.getvalue().encode("utf-8")
    manifest = {"run_summary_sha256": sha256(summary_path), "figures": records,
                "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
                "source_session_records_checked": validation["records_checked"],
                "source_session_supported_records": validation["supported_observation_session_records"],
                "test_is_fresh_independent_evidence": False}
    if any(output_dir.iterdir()):
        manifest_path = output_dir / "export_manifest.json"
        if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8-sig")) != manifest:
            raise ValueError("Publication directory is nonempty or does not match this run; choose a new output directory")
        for name, content in payloads:
            if sha256(output_dir / name) != hashlib.sha256(content).hexdigest():
                raise ValueError(f"Existing publication figure changed: {name}")
        return len(records)
    for name, content in payloads:
        (output_dir / name).write_bytes(content)
    (output_dir / "export_manifest.csv").write_bytes(csv_bytes)
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8-sig")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = export_run_figures(args.run_dir, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Figure export failed: {error}\n")
    print(f"Captioned and verified {count} figures in {args.output_dir}")


if __name__ == "__main__":
    main()

