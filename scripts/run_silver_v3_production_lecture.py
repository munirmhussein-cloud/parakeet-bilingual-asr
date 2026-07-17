from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

VIEWS = (
    "whole",
    "canonical_20s",
    "context_10s_stride_5s",
    "local_2p5s_contiguous",
)


def run(command: list[str], cwd: Path) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Silver v3 construction for one lecture without evaluation or quality gates."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--lecture-id", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--silver-root", type=Path, required=True)
    parser.add_argument("--view-workers", type=int, default=12)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    silver = args.silver_root.resolve()
    output_dir = silver / "reconciled_fixed"
    output_prefix = f"{args.lecture_id}_silver_v3_fixed"
    silver.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable, "-m", "pipeline.silver_v3.prepare_views",
        "--lecture-id", args.lecture_id,
        "--audio", str(args.audio),
        "--output-root", str(silver),
    ], repo)

    for view in VIEWS:
        run([
            sys.executable, "-m", "pipeline.silver_v3.run_hosted_view",
            "--lecture-id", args.lecture_id,
            "--view", view,
            "--manifest", str(silver / "manifests" / f"{args.lecture_id}_{view}.jsonl"),
            "--output-dir", str(silver / "raw_parakeet" / view),
            "--report", str(silver / f"silver_v3_{view}_hosted_report.json"),
            "--workers", str(args.view_workers),
        ], repo)

    normalization_report = silver / "silver_v3_normalization_report.json"
    run([
        sys.executable, "-m", "pipeline.silver_v3.normalize_views",
        "--lecture-id", args.lecture_id,
        "--silver-root", str(silver),
        "--report", str(normalization_report),
    ], repo)

    normalized = silver / "normalized"
    canonical_manifest = silver / "manifests" / f"{args.lecture_id}_canonical_20s.jsonl"
    run([
        sys.executable,
        str(repo / "scripts" / "reconcile_silver_v3_lattice.py"),
        "--lecture-id", args.lecture_id,
        "--whole", str(normalized / f"{args.lecture_id}_whole_normalized.jsonl"),
        "--canonical", str(normalized / f"{args.lecture_id}_canonical_20s_normalized.jsonl"),
        "--context", str(normalized / f"{args.lecture_id}_context_10s_stride_5s_normalized.jsonl"),
        "--local", str(normalized / f"{args.lecture_id}_local_2p5s_contiguous_normalized.jsonl"),
        "--canonical-manifest", str(canonical_manifest),
        "--output-dir", str(output_dir),
        "--output-prefix", output_prefix,
        "--schema-version", "silver_v3_segment_level_v2",
        "--title", f"{args.lecture_id} Silver v3",
    ], repo)

    segment_path = output_dir / f"{output_prefix}_segment_level.jsonl"
    report_path = output_dir / f"{output_prefix}_production_run_report.json"
    if not segment_path.exists():
        raise FileNotFoundError(segment_path)

    report = {
        "schema_version": "silver_v3_production_run_report_v1",
        "lecture_id": args.lecture_id,
        "segment_jsonl": str(segment_path),
        "evaluation_performed": False,
        "validation_performed": False,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"completed": True, **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
