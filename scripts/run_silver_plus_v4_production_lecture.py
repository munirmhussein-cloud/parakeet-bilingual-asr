from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from run_silver_plus_v3_fifth_view_export_v2 import integrate as integrate_v3
from run_silver_plus_v4_fifth_view_export import (
    BUILT_ON_SPINE,
    HONORIFIC_EQUIVALENCE,
    LAYER_VERSION,
    apply_fix_b,
    git_head,
    provenance_rows,
    read_jsonl,
    serialize,
    sha256_file,
)
from integrate_silver_plus_v3_fifth_view import write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Silver+ v4 production integration without Gold evaluation or validation gates."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--silver-v3", type=Path, required=True)
    parser.add_argument("--azure-parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-align-seconds", type=float, default=0.85)
    args = parser.parse_args()

    if "silver_plus_v4" not in args.output_prefix:
        raise ValueError("output-prefix must use lecture_XXX_silver_plus_v4 naming")

    silver_rows = read_jsonl(args.silver_v3)
    azure_rows = read_jsonl(args.azure_parent)
    if len(silver_rows) != len(azure_rows):
        raise ValueError(
            f"Row count mismatch: silver_v3={len(silver_rows)} azure_parent={len(azure_rows)}"
        )

    commit = git_head(args.repo_root)
    config = {
        "layer_version": LAYER_VERSION,
        "built_on_spine": BUILT_ON_SPINE,
        "max_align_seconds": args.max_align_seconds,
        "honorific_equivalence": HONORIFIC_EQUIVALENCE,
        "mode": "production_no_gold_validation",
    }
    config_hash = hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    baseline_rows, _ = integrate_v3(
        silver_rows,
        azure_rows,
        args.max_align_seconds,
        commit,
        config_hash,
    )
    rows = [apply_fix_b(row) for row in baseline_rows]
    provenance = provenance_rows(rows)

    rerun_baseline, _ = integrate_v3(
        silver_rows,
        azure_rows,
        args.max_align_seconds,
        commit,
        config_hash,
    )
    rerun_rows = [apply_fix_b(row) for row in rerun_baseline]
    deterministic = hashlib.sha256(serialize(rows)).digest() == hashlib.sha256(serialize(rerun_rows)).digest()
    if not deterministic:
        raise RuntimeError("Silver+ v4 production rerun was not deterministic")

    spine_unchanged = all(
        row["silver_v3_text"] == baseline["silver_v3_text"]
        for row, baseline in zip(rows, baseline_rows)
    )
    if not spine_unchanged:
        raise RuntimeError("silver_v3_text changed during Silver+ v4 production integration")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_path = args.output_dir / f"{args.output_prefix}_segment_level.jsonl"
    provenance_path = args.output_dir / f"{args.output_prefix}_token_provenance.jsonl"
    json_path = args.output_dir / f"{args.output_prefix}.json"
    manifest_path = args.output_dir / "PACKAGE_MANIFEST.json"
    report_path = args.output_dir / f"{args.output_prefix}_run_report.json"
    package_path = args.output_dir / f"{args.output_prefix}.zip"

    write_jsonl(segment_path, rows)
    write_jsonl(provenance_path, provenance)
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "silver_plus_v4_fifth_view_export_v1",
                "layer_version": LAYER_VERSION,
                "built_on_spine": BUILT_ON_SPINE,
                "repository_commit": commit,
                "config_hash": config_hash,
                "segment_count": len(rows),
                "segments": rows,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "silver_plus_v4_production_package_manifest_v1",
        "layer_version": LAYER_VERSION,
        "built_on_spine": BUILT_ON_SPINE,
        "repository_commit": commit,
        "config_hash": config_hash,
        "segment_count": len(rows),
        "evaluation_performed": False,
        "validation_performed": False,
        "files": [],
    }
    for path in (segment_path, provenance_path, json_path):
        manifest["files"].append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = {
        "schema_version": "silver_plus_v4_production_run_report_v1",
        "layer_version": LAYER_VERSION,
        "built_on_spine": BUILT_ON_SPINE,
        "repository_commit": commit,
        "config_hash": config_hash,
        "segment_count": len(rows),
        "deterministic": deterministic,
        "silver_v3_text_byte_identical_to_spine": spine_unchanged,
        "evaluation_performed": False,
        "validation_performed": False,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in (segment_path, provenance_path, json_path, manifest_path, report_path):
            archive.write(path, arcname=path.name)

    print(
        json.dumps(
            {
                "completed": True,
                "evaluation_performed": False,
                "validation_performed": False,
                "segment_count": len(rows),
                "segment_jsonl": str(segment_path),
                "zip": str(package_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
