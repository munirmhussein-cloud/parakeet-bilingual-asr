#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_reconciliation_gradio_input import (
    generate_gradio_input,
)


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
) -> tuple[bool, int]:
    if not path.exists() or path.stat().st_size == 0:
        return False, 0

    try:
        document = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False, 0

    if document.get("schema_version") != (
        "gradio_reconciliation_input_v1"
    ):
        return False, 0

    items = document.get("items")

    if not isinstance(items, list):
        return False, 0

    if items:
        first_audio_id = items[0].get("audio_id")
        if first_audio_id != segment_id:
            return False, 0

    return True, len(items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bronze-ar-dir", required=True)
    parser.add_argument("--bronze-en-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)

    if args.limit is not None:
        rows = rows[:args.limit]

    bronze_ar_dir = Path(args.bronze_ar_dir)
    bronze_en_dir = Path(args.bronze_en_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped_existing = 0
    failed: list[dict[str, Any]] = []
    total_items = 0

    started = time.perf_counter()

    for index, row in enumerate(rows, start=1):
        segment_id = row["segment_id"]
        audio_path = row["audio_filepath"]
        safe_segment_id = safe_name(segment_id)

        bronze_ar = (
            bronze_ar_dir
            / f"{safe_segment_id}.json"
        )
        bronze_en = (
            bronze_en_dir
            / f"{safe_segment_id}.json"
        )
        output = (
            output_dir
            / f"{safe_segment_id}_reconciliation.json"
        )

        if not args.force:
            valid, item_count = valid_existing_output(
                output,
                segment_id=segment_id,
            )

            if valid:
                skipped_existing += 1
                total_items += item_count

                print(
                    f"[{index}/{len(rows)}] "
                    f"skipped_existing: {segment_id}",
                    flush=True,
                )
                continue

        if not bronze_ar.exists() or not bronze_en.exists():
            failed.append({
                "segment_id": segment_id,
                "error": "missing_bronze_file",
                "bronze_ar_exists": bronze_ar.exists(),
                "bronze_en_exists": bronze_en.exists(),
            })

            print(
                f"[{index}/{len(rows)}] "
                f"failed_missing_bronze: {segment_id}",
                flush=True,
            )
            continue

        try:
            document = generate_gradio_input(
                bronze_ar_path=bronze_ar,
                bronze_en_path=bronze_en,
                output_path=output,
                audio_id=segment_id,
                audio_path=audio_path,
            )

            item_count = len(document.get("items", []))
            completed += 1
            total_items += item_count

            print(
                f"[{index}/{len(rows)}] "
                f"completed: {segment_id} "
                f"({item_count} rows)",
                flush=True,
            )

        except Exception as exc:
            failed.append({
                "segment_id": segment_id,
                "error": "reconciliation_failed",
                "exception": repr(exc),
            })

            print(
                f"[{index}/{len(rows)}] "
                f"failed: {segment_id}",
                flush=True,
            )

    elapsed = time.perf_counter() - started

    summary = {
        "manifest": args.manifest,
        "bronze_ar_dir": str(bronze_ar_dir),
        "bronze_en_dir": str(bronze_en_dir),
        "output_dir": str(output_dir),
        "attempted": len(rows),
        "completed": completed,
        "skipped_existing": skipped_existing,
        "failed": len(failed),
        "total_items": total_items,
        "elapsed_seconds": round(elapsed, 3),
        "failures": failed,
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
                    "attempted",
                    "completed",
                    "skipped_existing",
                    "failed",
                    "total_items",
                    "elapsed_seconds",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
