from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_step(name: str, command: list[str], cwd: Path) -> dict:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    record = {
        "name": name,
        "command": command,
        "returncode": result.returncode,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
    }
    print("=" * 100)
    print(name)
    print("=" * 100)
    print(result.stdout or "<empty stdout>")
    if result.stderr:
        print("STDERR")
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with return code {result.returncode}")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize repaired Silver v3 from existing raw hosted view results.")
    parser.add_argument("--lecture-id", required=True)
    parser.add_argument("--silver-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-prefix")
    parser.add_argument("--title")
    parser.add_argument("--schema-version", default="silver_v3_segment_level_v2")
    parser.add_argument("--min-a-tier-ratio", type=float, default=0.60)
    parser.add_argument("--max-repeated-6grams", type=int, default=25)
    parser.add_argument("--pilot-segments", type=int, default=30)
    parser.add_argument("--pilot-min-a-tier-ratio", type=float, default=0.75)
    parser.add_argument("--pilot-max-repeated-6grams", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    silver_root = args.silver_root.resolve()
    output_dir = (args.output_dir or silver_root / "reconciled_fixed").resolve()
    output_prefix = args.output_prefix or f"{args.lecture_id}_silver_v3_fixed"
    title = args.title or f"{args.lecture_id} Silver v3 — Repaired"
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_root = silver_root / "normalized"
    normalization_report = silver_root / "silver_v3_normalization_report.json"
    canonical_manifest = silver_root / "manifests" / f"{args.lecture_id}_canonical_20s.jsonl"
    segment_jsonl = output_dir / f"{output_prefix}_segment_level.jsonl"
    quality_report = output_dir / f"{output_prefix}_quality_report.json"
    run_report = output_dir / f"{output_prefix}_finalization_report.json"

    required = [
        canonical_manifest,
        silver_root / "manifests" / f"{args.lecture_id}_whole.jsonl",
        silver_root / "manifests" / f"{args.lecture_id}_context_10s_stride_5s.jsonl",
        silver_root / "manifests" / f"{args.lecture_id}_local_2p5s_contiguous.jsonl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Silver v3 inputs:\n" + "\n".join(missing))

    steps = []
    started = time.perf_counter()
    try:
        steps.append(run_step(
            "1. Normalize hosted views with timing repair",
            [
                sys.executable,
                "-m",
                "pipeline.silver_v3.normalize_views",
                "--lecture-id",
                args.lecture_id,
                "--silver-root",
                str(silver_root),
                "--report",
                str(normalization_report),
            ],
            repo_root,
        ))

        steps.append(run_step(
            "2. Reconcile repaired Silver v3 lattice",
            [
                sys.executable,
                str(repo_root / "scripts" / "reconcile_silver_v3_lattice.py"),
                "--lecture-id",
                args.lecture_id,
                "--whole",
                str(normalized_root / f"{args.lecture_id}_whole_normalized.jsonl"),
                "--canonical",
                str(normalized_root / f"{args.lecture_id}_canonical_20s_normalized.jsonl"),
                "--context",
                str(normalized_root / f"{args.lecture_id}_context_10s_stride_5s_normalized.jsonl"),
                "--local",
                str(normalized_root / f"{args.lecture_id}_local_2p5s_contiguous_normalized.jsonl"),
                "--canonical-manifest",
                str(canonical_manifest),
                "--output-dir",
                str(output_dir),
                "--output-prefix",
                output_prefix,
                "--schema-version",
                args.schema_version,
                "--title",
                title,
            ],
            repo_root,
        ))

        steps.append(run_step(
            "3. Apply Silver v3 production quality gates",
            [
                sys.executable,
                "-m",
                "pipeline.silver_v3.validate_quality",
                "--segment-jsonl",
                str(segment_jsonl),
                "--normalization-report",
                str(normalization_report),
                "--output",
                str(quality_report),
                "--min-a-tier-ratio",
                str(args.min_a_tier_ratio),
                "--max-repeated-6grams",
                str(args.max_repeated_6grams),
                "--pilot-segments",
                str(args.pilot_segments),
                "--pilot-min-a-tier-ratio",
                str(args.pilot_min_a_tier_ratio),
                "--pilot-max-repeated-6grams",
                str(args.pilot_max_repeated_6grams),
            ],
            repo_root,
        ))

        passed = True
        error = None
    except Exception as exc:
        passed = False
        error = repr(exc)

    report = {
        "schema_version": "silver_v3_fixed_finalization_report_v1",
        "lecture_id": args.lecture_id,
        "silver_root": str(silver_root),
        "output_dir": str(output_dir),
        "output_prefix": output_prefix,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "steps": steps,
        "normalization_report": str(normalization_report),
        "quality_report": str(quality_report),
        "segment_jsonl": str(segment_jsonl),
        "error": error,
        "passed": passed,
    }
    run_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
