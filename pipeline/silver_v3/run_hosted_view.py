from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one resumable hosted Parakeet Silver v3 view."
        )
    )

    parser.add_argument("--lecture-id", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retry-delays", default="5,15,30")
    parser.add_argument("--force", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)

    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("NVIDIA_API_KEY is not configured.")

    manifest_rows = read_jsonl(args.manifest)
    expected_ids = {
        str(row["segment_id"])
        for row in manifest_rows
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "scripts/run_bronze_inference_manifest.py",
        "--manifest",
        str(args.manifest),
        "--language",
        args.language,
        "--output-dir",
        str(args.output_dir),
        "--workers",
        str(args.workers),
        "--retry-delays",
        args.retry_delays,
    ]

    if args.force:
        command.append("--force")

    started = time.perf_counter()

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
    )

    valid_ids = set()
    total_words = 0
    empty_outputs = []
    invalid_outputs = []

    for path in sorted(args.output_dir.glob("*.json")):
        if path.name == "_summary.json":
            continue

        try:
            document = json.loads(
                path.read_text(encoding="utf-8")
            )
        except Exception as error:
            invalid_outputs.append(
                {
                    "path": str(path),
                    "error": repr(error),
                }
            )
            continue

        audio_id = str(document.get("audio_id", ""))
        words = document.get("words")

        if (
            audio_id not in expected_ids
            or document.get("language") != args.language
            or not isinstance(words, list)
        ):
            invalid_outputs.append(
                {
                    "path": str(path),
                    "error": "schema_or_identity_mismatch",
                }
            )
            continue

        valid_ids.add(audio_id)
        total_words += len(words)

        if not words:
            empty_outputs.append(audio_id)

    missing_ids = sorted(expected_ids - valid_ids)

    passed = (
        result.returncode == 0
        and not invalid_outputs
        and not missing_ids
        and valid_ids == expected_ids
    )

    report = {
        "schema_version":
        "silver_v3_parakeet_view_timing_v1",
        "lecture_id":
        args.lecture_id,
        "view":
        args.view,
        "language":
        args.language,
        "workers":
        args.workers,
        "wall_seconds":
        round(time.perf_counter() - started, 3),
        "return_code":
        result.returncode,
        "expected_count":
        len(expected_ids),
        "valid_output_count":
        len(valid_ids),
        "missing_output_count":
        len(missing_ids),
        "invalid_output_count":
        len(invalid_outputs),
        "total_word_count":
        total_words,
        "empty_word_output_count":
        len(empty_outputs),
        "empty_word_outputs":
        empty_outputs,
        "missing_ids":
        missing_ids,
        "invalid_outputs":
        invalid_outputs,
        "stdout":
        result.stdout,
        "stderr":
        result.stderr,
        "passed":
        passed,
    }

    args.report.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr, file=sys.stderr)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
