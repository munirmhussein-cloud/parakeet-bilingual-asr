#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("–", "-")
    )


def valid_existing_output(
    path: Path,
    *,
    segment_id: str,
    language: str,
) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        document = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False

    return (
        document.get("audio_id") == segment_id
        and document.get("language") == language
        and isinstance(document.get("words"), list)
    )


def run_one(
    row: dict[str, Any],
    *,
    language: str,
    output_dir: Path,
    force: bool,
) -> dict[str, Any]:
    audio_path = row["audio_filepath"]
    segment_id = row["segment_id"]

    output_path = (
        output_dir
        / f"{safe_name(segment_id)}.json"
    )

    if not force and valid_existing_output(
        output_path,
        segment_id=segment_id,
        language=language,
    ):
        return {
            "segment_id": segment_id,
            "audio_filepath": audio_path,
            "output": str(output_path),
            "status": "skipped_existing",
            "returncode": 0,
        }

    command = [
        sys.executable,
        "scripts/run_bronze_inference.py",
        "--audio",
        audio_path,
        "--output",
        str(output_path),
        "--language",
        language,
        "--audio-id",
        segment_id,
    ]

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    return {
        "segment_id": segment_id,
        "audio_filepath": audio_path,
        "output": str(output_path),
        "status": (
            "completed"
            if result.returncode == 0
            else "failed"
        ),
        "returncode": result.returncode,
        "output_tail": result.stdout[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    rows = read_jsonl(args.manifest)

    if args.limit is not None:
        rows = rows[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    total = len(rows)

    print(
        f"Starting {args.language} inference: "
        f"{total} segments, {args.workers} workers.",
        flush=True,
    )

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        future_map = {
            executor.submit(
                run_one,
                row,
                language=args.language,
                output_dir=output_dir,
                force=args.force,
            ): row
            for row in rows
        }

        for completed_index, future in enumerate(
            as_completed(future_map),
            start=1,
        ):
            row = future_map[future]

            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "segment_id": row.get("segment_id"),
                    "audio_filepath": row.get(
                        "audio_filepath"
                    ),
                    "status": "failed",
                    "returncode": 1,
                    "error": repr(exc),
                }

            results.append(result)

            print(
                f"[{completed_index}/{total}] "
                f"{result['status']}: "
                f"{result.get('segment_id')}",
                flush=True,
            )

    failures = [
        result
        for result in results
        if result["status"] == "failed"
    ]

    summary = {
        "language": args.language,
        "manifest": args.manifest,
        "output_dir": str(output_dir),
        "attempted": total,
        "completed": sum(
            result["status"] == "completed"
            for result in results
        ),
        "skipped_existing": sum(
            result["status"] == "skipped_existing"
            for result in results
        ),
        "failed": len(failures),
        "workers": args.workers,
        "failures": failures,
    }

    summary_path = output_dir / "_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(summary, ensure_ascii=False, indent=2),
        flush=True,
    )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
