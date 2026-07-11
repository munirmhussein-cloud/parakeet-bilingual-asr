#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.parakeet_infer import RivaTranscriber


REPO_ROOT = Path(__file__).resolve().parents[1]

TRANSIENT_ERROR_MARKERS = (
    "StatusCode.UNAVAILABLE",
    "StatusCode.DEADLINE_EXCEEDED",
    "StatusCode.RESOURCE_EXHAUSTED",
    "Received http2 header with status: 502",
    "Received http2 header with status: 503",
    "connection reset",
    "Connection reset",
    "temporarily unavailable",
    "Temporary failure",
)

DEFAULT_RETRY_DELAYS = (2.0, 5.0, 15.0)

_THREAD_STATE = local()


def get_transcriber(language: str) -> RivaTranscriber:
    transcriber = getattr(_THREAD_STATE, "transcriber", None)
    transcriber_language = getattr(
        _THREAD_STATE,
        "transcriber_language",
        None,
    )

    if (
        transcriber is None
        or transcriber_language != language
    ):
        transcriber = RivaTranscriber(
            language=language,
        )
        _THREAD_STATE.transcriber = transcriber
        _THREAD_STATE.transcriber_language = language

    return transcriber


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


def is_transient_failure(output: str) -> bool:
    return any(
        marker in output
        for marker in TRANSIENT_ERROR_MARKERS
    )


def run_one(
    row: dict[str, Any],
    *,
    language: str,
    output_dir: Path,
    force: bool,
    retry_delays: tuple[float, ...],
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
            "attempt_count": 0,
            "transient_errors": [],
        }

    attempts: list[dict[str, Any]] = []
    maximum_attempts = 1 + len(retry_delays)

    for attempt_number in range(1, maximum_attempts + 1):
        started = time.perf_counter()

        try:
            transcriber = get_transcriber(language)

            document = transcriber.transcribe(
                audio_path,
                audio_id=segment_id,
            )

            output_path.write_text(
                json.dumps(
                    document,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            elapsed = time.perf_counter() - started

            attempts.append({
                "attempt": attempt_number,
                "returncode": 0,
                "elapsed_seconds": round(elapsed, 3),
                "transient": False,
            })

            return {
                "segment_id": segment_id,
                "audio_filepath": audio_path,
                "output": str(output_path),
                "status": "completed",
                "returncode": 0,
                "attempt_count": attempt_number,
                "retry_count": attempt_number - 1,
                "attempts": attempts,
            }

        except Exception as exc:
            elapsed = time.perf_counter() - started
            output = repr(exc)
            transient = is_transient_failure(output)

            attempts.append({
                "attempt": attempt_number,
                "returncode": 1,
                "elapsed_seconds": round(elapsed, 3),
                "transient": transient,
                "error": output,
            })

            # Recreate this worker's channel on the next retry.
            _THREAD_STATE.transcriber = None
            _THREAD_STATE.transcriber_language = None

            if not transient:
                break

            if attempt_number <= len(retry_delays):
                delay = retry_delays[attempt_number - 1]
                time.sleep(delay)

    return {
        "segment_id": segment_id,
        "audio_filepath": audio_path,
        "output": str(output_path),
        "status": "failed",
        "returncode": attempts[-1]["returncode"],
        "attempt_count": len(attempts),
        "retry_count": max(0, len(attempts) - 1),
        "attempts": attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--retry-delays",
        default="2,5,15",
        help=(
            "Comma-separated retry delays in seconds for transient "
            "endpoint failures. Empty string disables retries."
        ),
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    retry_delays = tuple(
        float(value)
        for value in args.retry_delays.split(",")
        if value.strip()
    )

    rows = read_jsonl(args.manifest)

    if args.limit is not None:
        rows = rows[:args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    total = len(rows)

    print(
        f"Starting {args.language} inference: "
        f"{total} segments, {args.workers} workers, "
        f"{len(retry_delays)} retries.",
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
                retry_delays=retry_delays,
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
                    "attempt_count": 0,
                    "retry_count": 0,
                    "error": repr(exc),
                }

            results.append(result)

            retry_suffix = (
                f", retries={result.get('retry_count', 0)}"
                if result.get("retry_count")
                else ""
            )

            print(
                f"[{completed_index}/{total}] "
                f"{result['status']}: "
                f"{result.get('segment_id')}"
                f"{retry_suffix}",
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
        "retry_delays": retry_delays,
        "segments_retried": sum(
            result.get("retry_count", 0) > 0
            for result in results
        ),
        "total_retries": sum(
            result.get("retry_count", 0)
            for result in results
        ),
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
        json.dumps(
            {
                key: summary[key]
                for key in [
                    "language",
                    "attempted",
                    "completed",
                    "skipped_existing",
                    "failed",
                    "workers",
                    "segments_retried",
                    "total_retries",
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
