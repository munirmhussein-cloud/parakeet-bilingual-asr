#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def run(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run audio preparation, bilingual Bronze inference, "
            "reconciliation, automatic language tagging, and SILVER "
            "JSONL export."
        )
    )
    parser.add_argument("--audio", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--source-audio-id")
    parser.add_argument("--segment-seconds", type=float, default=20.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent Riva inference workers per language.",
    )
    parser.add_argument(
        "--force-inference",
        action="store_true",
        help="Re-run Bronze outputs even when valid files exist.",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    workspace = Path(args.workspace).resolve()

    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError(
            "NVIDIA_API_KEY is not configured in this process."
        )

    source_audio_id = (
        args.source_audio_id
        or slugify(audio_path.stem)
    )

    if not source_audio_id:
        raise ValueError(
            "Unable to derive source_audio_id; provide "
            "--source-audio-id explicitly."
        )

    paths = {
        "catalog": (
            workspace
            / "data/catalogs"
            / f"{source_audio_id}_audio_catalog.json"
        ),
        "segments": (
            workspace
            / "data/segments"
            / source_audio_id
        ),
        "metadata": (
            workspace
            / "data/segment_metadata"
            / f"{source_audio_id}_segments.json"
        ),
        "manifest": (
            workspace
            / "data/manifests"
            / f"{source_audio_id}_segments.jsonl"
        ),
        "segment_validation": (
            workspace
            / "data/validation"
            / f"{source_audio_id}_segment_validation.json"
        ),
        "bronze_en": (
            workspace
            / "data/bronze/en"
            / source_audio_id
        ),
        "bronze_ar": (
            workspace
            / "data/bronze/ar"
            / source_audio_id
        ),
        "reconciliation": (
            workspace
            / "data/reconciliation"
            / source_audio_id
        ),
        "silver_jsonl": (
            workspace
            / "data/annotations/silver"
            / source_audio_id
            / f"{source_audio_id}_silver.jsonl"
        ),
        "silver_report": (
            workspace
            / "data/validation"
            / f"{source_audio_id}_silver_report.json"
        ),
    }

    for key, path in paths.items():
        if key in {
            "segments",
            "bronze_en",
            "bronze_ar",
            "reconciliation",
        }:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

    prep_outputs_exist = all(
        paths[key].exists()
        for key in [
            "catalog",
            "metadata",
            "manifest",
            "segment_validation",
        ]
    )

    if args.force or not prep_outputs_exist:
        run([
            sys.executable,
            "scripts/run_sprint4_audio_prep.py",
            "--input",
            str(audio_path),
            "--source-audio-id",
            source_audio_id,
            "--catalog-out",
            str(paths["catalog"]),
            "--output-dir",
            str(paths["segments"]),
            "--metadata-out",
            str(paths["metadata"]),
            "--manifest-out",
            str(paths["manifest"]),
            "--validation-report-out",
            str(paths["segment_validation"]),
            "--segment-seconds",
            str(args.segment_seconds),
            "--sample-rate",
            str(args.sample_rate),
            "--notes",
            "Automatic SILVER pipeline",
        ])
    else:
        print("\nSkipping audio preparation: outputs already exist.")

    run([
        sys.executable,
        "scripts/run_bronze_inference_manifest.py",
        "--manifest",
        str(paths["manifest"]),
        "--language",
        "en-US",
        "--output-dir",
        str(paths["bronze_en"]),
        "--workers",
        str(args.workers),
        *(
            ["--force"]
            if args.force_inference
            else []
        ),
    ])

    run([
        sys.executable,
        "scripts/run_bronze_inference_manifest.py",
        "--manifest",
        str(paths["manifest"]),
        "--language",
        "ar-AR",
        "--output-dir",
        str(paths["bronze_ar"]),
        "--workers",
        str(args.workers),
        *(
            ["--force"]
            if args.force_inference
            else []
        ),
    ])

    run([
        sys.executable,
        "scripts/run_reconciliation_manifest.py",
        "--manifest",
        str(paths["manifest"]),
        "--bronze-ar-dir",
        str(paths["bronze_ar"]),
        "--bronze-en-dir",
        str(paths["bronze_en"]),
        "--output-dir",
        str(paths["reconciliation"]),
    ])

    run([
        sys.executable,
        "scripts/export_reconciliation_to_silver.py",
        "--manifest",
        str(paths["manifest"]),
        "--reconciliation-dir",
        str(paths["reconciliation"]),
        "--output-jsonl",
        str(paths["silver_jsonl"]),
        "--report",
        str(paths["silver_report"]),
    ])

    report = json.loads(
        paths["silver_report"].read_text(encoding="utf-8")
    )

    print("\nSILVER pipeline completed.")
    print("Source audio ID:", source_audio_id)
    print("Manifest:", paths["manifest"])
    print("SILVER JSONL:", paths["silver_jsonl"])
    print("SILVER report:", paths["silver_report"])
    print("Exported segments:", report["exported_segments"])
    print("Exported rows:", report["exported_rows"])
    print("Empty segments:", report["empty_segment_count"])


if __name__ == "__main__":
    main()
