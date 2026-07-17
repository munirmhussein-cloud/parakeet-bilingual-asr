from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print()
    print("$", " ".join(command))

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
    )

    if result.returncode != 0:
        raise SystemExit(
            result.returncode
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize Silver v3 from completed "
            "hosted Parakeet multiview outputs."
        )
    )

    parser.add_argument(
        "--lecture-id",
        required=True,
    )

    parser.add_argument(
        "--lecture-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--title",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    silver_root = (
        args.lecture_root
        / "silver_v3"
    )

    normalized = (
        silver_root
        / "normalized"
    )

    manifests = (
        silver_root
        / "manifests"
    )

    reconciled = (
        silver_root
        / "reconciled"
    )

    runtime = (
        args.lecture_root
        / "runtime"
    )

    runtime.mkdir(
        parents=True,
        exist_ok=True,
    )

    run(
        [
            sys.executable,
            "-m",
            "pipeline.silver_v3.normalize_views",

            "--lecture-id",
            args.lecture_id,

            "--silver-root",
            str(silver_root),

            "--report",
            str(
                runtime
                / "silver_v3_normalization_report.json"
            ),
        ]
    )

    output_prefix = (
        f"{args.lecture_id}_silver_v3"
    )

    run(
        [
            sys.executable,
            "scripts/reconcile_silver_v3_lattice.py",

            "--lecture-id",
            args.lecture_id,

            "--whole",
            str(
                normalized
                / (
                    f"{args.lecture_id}"
                    "_whole_normalized.jsonl"
                )
            ),

            "--canonical",
            str(
                normalized
                / (
                    f"{args.lecture_id}"
                    "_canonical_20s_normalized.jsonl"
                )
            ),

            "--context",
            str(
                normalized
                / (
                    f"{args.lecture_id}"
                    "_context_10s_stride_5s"
                    "_normalized.jsonl"
                )
            ),

            "--local",
            str(
                normalized
                / (
                    f"{args.lecture_id}"
                    "_local_2p5s_contiguous"
                    "_normalized.jsonl"
                )
            ),

            "--canonical-manifest",
            str(
                manifests
                / (
                    f"{args.lecture_id}"
                    "_canonical_manifest.jsonl"
                )
            ),

            "--output-dir",
            str(reconciled),

            "--output-prefix",
            output_prefix,

            "--schema-version",
            "silver_v3_segment_level_v1",

            "--title",
            (
                args.title
                or f"{args.lecture_id} — Silver v3"
            ),
        ]
    )

    report_path = (
        reconciled
        / f"{output_prefix}_report.json"
    )

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    if not report["validation"]["passed"]:
        raise RuntimeError(
            json.dumps(
                report["validation"],
                indent=2,
                sort_keys=True,
            )
        )

    print()
    print("=" * 80)
    print("SILVER V3 COMPLETE")
    print("=" * 80)
    print("Lecture:", args.lecture_id)
    print(
        "Segments:",
        report["segment_count"],
    )
    print(
        "Tokens:",
        report["total_tokens"],
    )
    print(
        "Validation:",
        report["validation"],
    )
    print(
        "Report:",
        report_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
