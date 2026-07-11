#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.annotation.automatic_language import (
    assign_languages_with_inheritance,
)
from src.annotation.gradio_data_model import (
    normalize_reconciliation_input,
)


SCHEMA_VERSION = "silver_annotations_v1"


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def segment_number(segment_id: str) -> str | None:
    match = re.search(r"seg_\d{6}", segment_id)
    return match.group(0) if match else None


def find_reconciliation(
    reconciliation_dir: Path,
    segment_id: str,
) -> Path:
    number = segment_number(segment_id)
    candidates: list[Path] = []

    if number:
        candidates.extend(
            reconciliation_dir.glob(
                f"*{number}*_reconciliation.json"
            )
        )

    if not candidates:
        raise FileNotFoundError(
            f"No reconciliation file for segment: {segment_id}"
        )

    return sorted(candidates)[0]


def to_silver_item(
    item: dict[str, Any],
    *,
    segment_id: str,
    audio_filepath: str,
) -> dict[str, Any]:
    training_span = item.get("training_span") or {}

    return {
        "schema_version": SCHEMA_VERSION,
        "segment_id": segment_id,
        "audio_filepath": audio_filepath,
        "row_id": item["row_id"],
        "audio_id": item["audio_id"],
        "word_index": item["word_index"],
        "global_start": item["global_start"],
        "global_end": item["global_end"],
        "local_start": item.get("local_start"),
        "local_end": item.get("local_end"),
        "selected_language": item["selected_language"],
        "corrected_text": item["corrected_text"],
        "review_status": "auto_reviewed",
        "reviewed_at": None,
        "reviewer_id": None,
        "language_assignment_method": item.get(
            "language_assignment_method"
        ),
        "language_assignment_source": item.get(
            "language_assignment_source"
        ),
        "reconciliation_flags": item.get(
            "reconciliation_flags",
            [],
        ),
        "bronze": {
            "bronze_ar_text": item.get(
                "bronze_ar_text",
                "",
            ),
            "bronze_en_text": item.get(
                "bronze_en_text",
                "",
            ),
            "original_selected_language": item.get(
                "original_selected_language"
            ),
            "original_corrected_text": item.get(
                "original_corrected_text"
            ),
            "language_hypotheses": item.get(
                "language_hypotheses",
                [],
            ),
        },
        "context": {
            "left_context": item.get("left_context", []),
            "right_context": item.get("right_context", []),
        },
        "training_span": {
            "span_id": training_span.get("span_id"),
            "language": item["selected_language"],
            "start": training_span.get(
                "start",
                item.get("global_start"),
            ),
            "end": training_span.get(
                "end",
                item.get("global_end"),
            ),
            "text": item["corrected_text"],
        },
        "source": item.get("source", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate machine-tagged SILVER JSONL from a segment "
            "manifest and reconciliation directory."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reconciliation-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    reconciliation_dir = Path(args.reconciliation_dir)
    output_jsonl = Path(args.output_jsonl)
    report_path = Path(args.report)

    manifest_rows = read_jsonl(manifest_path)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    output_items: list[dict[str, Any]] = []
    empty_segments: list[dict[str, Any]] = []
    segment_summaries: list[dict[str, Any]] = []

    for row in manifest_rows:
        segment_id = row["segment_id"]
        audio_filepath = row["audio_filepath"]

        reconciliation_path = find_reconciliation(
            reconciliation_dir,
            segment_id,
        )

        normalized = normalize_reconciliation_input(
            reconciliation_path
        )

        items = normalized.get("items", [])

        if not items:
            empty_segments.append(
                {
                    "segment_id": segment_id,
                    "audio_filepath": audio_filepath,
                    "duration": row.get("duration"),
                    "reconciliation_path": str(
                        reconciliation_path
                    ),
                    "reason": "zero_reconciliation_items",
                }
            )
            continue

        tagged_items = assign_languages_with_inheritance(items)

        silver_items = [
            to_silver_item(
                item,
                segment_id=segment_id,
                audio_filepath=audio_filepath,
            )
            for item in tagged_items
        ]

        output_items.extend(silver_items)

        segment_summaries.append(
            {
                "segment_id": segment_id,
                "audio_filepath": audio_filepath,
                "row_count": len(silver_items),
                "language_counts": dict(
                    Counter(
                        item["selected_language"]
                        for item in silver_items
                    )
                ),
                "assignment_method_counts": dict(
                    Counter(
                        item["language_assignment_method"]
                        for item in silver_items
                    )
                ),
            }
        )

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for item in output_items:
            handle.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    report = {
        "schema_version": "silver_export_report_v1",
        "created_at": utc_now_iso(),
        "manifest": str(manifest_path),
        "reconciliation_dir": str(reconciliation_dir),
        "output_jsonl": str(output_jsonl),
        "manifest_segments": len(manifest_rows),
        "exported_segments": len(segment_summaries),
        "empty_segment_count": len(empty_segments),
        "exported_rows": len(output_items),
        "language_counts": dict(
            Counter(
                item["selected_language"]
                for item in output_items
            )
        ),
        "assignment_method_counts": dict(
            Counter(
                item["language_assignment_method"]
                for item in output_items
            )
        ),
        "empty_segments": empty_segments,
        "segments": segment_summaries,
    }

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
