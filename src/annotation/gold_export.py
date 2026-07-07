"""
Sprint 2 Gold export utilities.

Exports reviewed word-level ReviewItem dictionaries into Gold-ready JSON
and JSONL while preserving Bronze lineage and future NeMo span fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .gradio_data_model import utc_now_iso


SCHEMA_VERSION = "gold_annotations_v1"


def summarize_gold_items(items: list[dict[str, Any]]) -> dict[str, int]:
    reviewed = [item for item in items if item.get("review_status") == "reviewed"]

    return {
        "total_rows": len(items),
        "reviewed_rows": len(reviewed),
        "flagged_rows": sum(1 for item in items if item.get("reconciliation_flags")),
        "language_changed_rows": sum(
            1
            for item in reviewed
            if item.get("selected_language") != item.get("original_selected_language")
        ),
        "text_changed_rows": sum(
            1
            for item in reviewed
            if item.get("corrected_text") != item.get("original_corrected_text")
        ),
    }


def to_gold_item(item: dict[str, Any]) -> dict[str, Any]:
    training_span = item.get("training_span") or {}

    return {
        "row_id": item["row_id"],
        "audio_id": item["audio_id"],
        "word_index": item["word_index"],
        "global_start": item["global_start"],
        "global_end": item["global_end"],
        "local_start": item.get("local_start"),
        "local_end": item.get("local_end"),
        "selected_language": item["selected_language"],
        "corrected_text": item["corrected_text"],
        "review_status": item.get("review_status", "unreviewed"),
        "reviewed_at": item.get("reviewed_at"),
        "reviewer_id": item.get("reviewer_id"),
        "reconciliation_flags": item.get("reconciliation_flags", []),
        "bronze": {
            "bronze_ar_text": item.get("bronze_ar_text", ""),
            "bronze_en_text": item.get("bronze_en_text", ""),
            "original_selected_language": item.get("original_selected_language"),
            "original_corrected_text": item.get("original_corrected_text"),
            "language_hypotheses": item.get("language_hypotheses", []),
        },
        "context": {
            "left_context": item.get("left_context", []),
            "right_context": item.get("right_context", []),
        },
        "training_span": {
            "span_id": training_span.get("span_id"),
            "language": item["selected_language"],
            "start": training_span.get("start", item.get("global_start")),
            "end": training_span.get("end", item.get("global_end")),
            "text": item["corrected_text"],
        },
        "source": item.get("source", {}),
    }


def export_gold_annotations(
    items: list[dict[str, Any]],
    output_json_path: str | Path = "annotations/gold/gold_annotations_v1.json",
    output_jsonl_path: str | Path | None = "annotations/gold/gold_annotations_v1.jsonl",
    source_file: str | None = None,
) -> dict[str, Any]:
    gold_items = [to_gold_item(item) for item in items]

    document = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "source_file": source_file,
        "review_summary": summarize_gold_items(items),
        "items": gold_items,
    }

    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if output_jsonl_path is not None:
        output_jsonl_path = Path(output_jsonl_path)
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl_path.open("w", encoding="utf-8") as handle:
            for item in gold_items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    return document
