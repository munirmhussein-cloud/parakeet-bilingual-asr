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
    parser = argparse.ArgumentParser(description="Run accepted Silver v3 production workflow for one lecture.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--lecture-id", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--silver-root", type=Path, required=True)
    parser.add_argument("--view-workers", type=int, default=12)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    silver = args.silver_root.resolve()
    silver.mkdir(parents=True, exist_ok=True)

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

    run([
        sys.executable, "-m", "pipeline.silver_v3.finalize_fixed",
        "--lecture-id", args.lecture_id,
        "--silver-root", str(silver),
        "--repo-root", str(repo),
        "--output-prefix", f"{args.lecture_id}_silver_v3_fixed",
    ], repo)

    expected = silver / "reconciled_fixed" / f"{args.lecture_id}_silver_v3_fixed_segment_level.jsonl"
    if not expected.exists():
        raise FileNotFoundError(expected)

    print(json.dumps({"completed": True, "lecture_id": args.lecture_id, "segment_jsonl": str(expected)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
