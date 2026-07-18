#!/usr/bin/env python3
"""Export a completed Faster-Whisper whole JSON as Copper v1 canonical 20s outputs.

Copper v1 is the formal name for the pipeline previously developed as Bronze
v2.4. The exporter reuses the proven Bronze v2.4 canonicalization implementation,
then rewrites package metadata and filenames to the Copper v1 contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

COPPER_SCHEMA = "copper_v1_canonical_20s_v1"
COPPER_PIPELINE = "copper_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lecture-id", required=True)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--whole-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rewrite_row(row: dict) -> dict:
    updated = dict(row)
    updated["schema_version"] = COPPER_SCHEMA
    updated["pipeline_version"] = COPPER_PIPELINE
    if "bronze_text" in updated:
        updated["copper_text"] = updated.pop("bronze_text")
    if "has_bronze_text" in updated:
        updated["has_copper_text"] = updated.pop("has_bronze_text")
    return updated


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    legacy_exporter = Path(__file__).with_name("export_bronze_v2_4_canonical.py")
    if not legacy_exporter.is_file():
        raise FileNotFoundError(legacy_exporter)

    command = [
        sys.executable,
        str(legacy_exporter),
        "--lecture-id",
        args.lecture_id,
        "--audio",
        str(args.audio.expanduser().resolve()),
        "--whole-json",
        str(args.whole_json.expanduser().resolve()),
        "--output-dir",
        str(output_dir),
        "--overwrite",
    ]
    subprocess.run(command, check=True)

    legacy_json = output_dir / f"{args.lecture_id}_bronze_v2_4_canonical_20s.json"
    legacy_jsonl = output_dir / f"{args.lecture_id}_bronze_v2_4_canonical_20s.jsonl"
    legacy_docx = output_dir / f"{args.lecture_id}_bronze_v2_4_canonical_20s.docx"
    manifest_path = output_dir / "PACKAGE_MANIFEST.json"

    copper_json = output_dir / f"{args.lecture_id}_copper_v1_canonical_20s.json"
    copper_jsonl = output_dir / f"{args.lecture_id}_copper_v1_canonical_20s.jsonl"
    copper_docx = output_dir / f"{args.lecture_id}_copper_v1_canonical_20s.docx"

    package = json.loads(legacy_json.read_text(encoding="utf-8"))
    package["schema_version"] = COPPER_SCHEMA
    package["pipeline_version"] = COPPER_PIPELINE
    package["legacy_pipeline_name"] = "bronze_v2.4"
    package["segments"] = [rewrite_row(row) for row in package.get("segments", [])]
    package["text"] = " ".join(
        row.get("copper_text", "")
        for row in package["segments"]
        if row.get("copper_text")
    )
    copper_json.write_text(
        json.dumps(package, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with copper_jsonl.open("w", encoding="utf-8") as handle:
        for row in package["segments"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    legacy_docx.replace(copper_docx)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "copper_v1_package_manifest_v1"
    manifest["pipeline_version"] = COPPER_PIPELINE
    manifest["legacy_pipeline_name"] = "bronze_v2.4"
    manifest["files"] = {
        "whole_json": args.whole_json.expanduser().resolve().name,
        "canonical_json": copper_json.name,
        "canonical_jsonl": copper_jsonl.name,
        "docx": copper_docx.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    legacy_json.unlink(missing_ok=True)
    legacy_jsonl.unlink(missing_ok=True)

    print("\n=== COPPER V1 CANONICAL EXPORT COMPLETE ===")
    print(f"JSON: {copper_json}")
    print(f"JSONL: {copper_jsonl}")
    print(f"DOCX: {copper_docx}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
