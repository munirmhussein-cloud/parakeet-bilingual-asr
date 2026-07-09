#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path


def load_config(path):
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required for --config. Install with: pip install pyyaml"
        ) from exc

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cmd):
    print("\nRunning:")
    print(" ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def get_arg(args, config, key, required=True, default=None):
    value = getattr(args, key, None)
    if value is not None:
        return value

    value = config.get(key) if config else None
    if value is not None:
        return value

    if required:
        raise SystemExit(f"Missing required argument/config value: {key}")

    return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    parser.add_argument("--input")
    parser.add_argument("--source-audio-id")
    parser.add_argument("--catalog-out")
    parser.add_argument("--output-dir")
    parser.add_argument("--metadata-out")
    parser.add_argument("--manifest-out")
    parser.add_argument("--validation-report-out")
    parser.add_argument("--segment-seconds", type=float)
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--notes")

    args = parser.parse_args()
    config = load_config(args.config) if args.config else {}

    input_path = get_arg(args, config, "input")
    source_audio_id = get_arg(args, config, "source_audio_id", required=False)
    catalog_out = get_arg(args, config, "catalog_out")
    output_dir = get_arg(args, config, "output_dir")
    metadata_out = get_arg(args, config, "metadata_out")
    manifest_out = get_arg(args, config, "manifest_out")
    validation_report_out = get_arg(args, config, "validation_report_out")
    segment_seconds = get_arg(args, config, "segment_seconds", required=False, default=20.0)
    sample_rate = get_arg(args, config, "sample_rate", required=False, default=16000)
    notes = get_arg(args, config, "notes", required=False, default="")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(catalog_out).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_out).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_out).parent.mkdir(parents=True, exist_ok=True)
    Path(validation_report_out).parent.mkdir(parents=True, exist_ok=True)

    catalog_cmd = [
        "python", "scripts/catalog_audio.py",
        "--input", input_path,
        "--catalog-out", catalog_out,
        "--notes", notes,
    ]

    if source_audio_id:
        catalog_cmd.extend(["--source-audio-id", source_audio_id])

    run(catalog_cmd)

    run([
        "python", "scripts/segment_audio.py",
        "--input", input_path,
        "--output-dir", output_dir,
        "--metadata-out", metadata_out,
        "--segment-seconds", str(segment_seconds),
        "--sample-rate", str(sample_rate),
    ])

    run([
        "python", "scripts/validate_segments.py",
        "--metadata", metadata_out,
        "--report-out", validation_report_out,
    ])

    run([
        "python", "scripts/export_segment_manifest.py",
        "--metadata", metadata_out,
        "--manifest-out", manifest_out,
    ])

    print("\nSprint 4 audio preparation completed successfully.")
    print(f"Catalog: {catalog_out}")
    print(f"Segments: {output_dir}")
    print(f"Metadata: {metadata_out}")
    print(f"Validation report: {validation_report_out}")
    print(f"Manifest: {manifest_out}")


if __name__ == "__main__":
    main()
