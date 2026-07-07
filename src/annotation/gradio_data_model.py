"""
Sprint 2 Gradio data model utilities.

Validates gradio_reconciliation_input_v1.json and normalizes raw
word-level reconciliation rows into stable ReviewItem dictionaries.
"""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_LANGUAGES = {"ar-AR", "en-US"}

REQUIRED_TOP_LEVEL = {"schema_version", "items"}

REQUIRED_ITEM_FIELDS = {
    "row_id",
    "audio_id",
    "word_index",
    "global_start",
    "global_end",
    "selected_language",
    "corrected_text",
}

OPTIONAL_WITH_DEFAULTS = {
    "local_start": None,
    "local_end": None,
    "bronze_ar_text": "",
    "bronze_en_text": "",
    "reconciliation_flags": [],
    "left_context": [],
    "right_context": [],
    "language_hypotheses": [],
    "review_status": "unreviewed",
    "reviewed_at": None,
    "reviewer_id": None,
    "training_span": None,
}


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_reconciliation_input(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []

    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    data = load_json(path)

    missing_top = REQUIRED_TOP_LEVEL - set(data.keys())
    if missing_top:
        errors.append(f"Missing top-level fields: {sorted(missing_top)}")

    items = data.get("items")
    if not isinstance(items, list):
        errors.append("Top-level `items` must be a list.")
        items = []

    seen_row_ids = set()

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"Item {idx} is not an object.")
            continue

        missing = REQUIRED_ITEM_FIELDS - set(item.keys())
        if missing:
            errors.append(f"Item {idx} missing required fields: {sorted(missing)}")

        for field in OPTIONAL_WITH_DEFAULTS:
            if field not in item:
                warnings.append(
                    f"Item {idx} missing optional field `{field}`; default can be added."
                )

        row_id = item.get("row_id")
        if row_id:
            if row_id in seen_row_ids:
                errors.append(f"Duplicate row_id: {row_id}")
            seen_row_ids.add(row_id)

        lang = item.get("selected_language")
        if lang not in VALID_LANGUAGES:
            errors.append(
                f"Item {idx} row_id={row_id} has invalid selected_language: {lang}"
            )

        if not isinstance(item.get("corrected_text"), str):
            errors.append(f"Item {idx} row_id={row_id} corrected_text must be a string.")

        for time_field in ["global_start", "global_end"]:
            value = item.get(time_field)
            if not isinstance(value, (int, float)):
                errors.append(f"Item {idx} row_id={row_id} {time_field} must be numeric.")

        if isinstance(item.get("global_start"), (int, float)) and isinstance(
            item.get("global_end"), (int, float)
        ):
            if item["global_end"] < item["global_start"]:
                errors.append(
                    f"Item {idx} row_id={row_id} has global_end before global_start."
                )

        for list_field in [
            "reconciliation_flags",
            "left_context",
            "right_context",
            "language_hypotheses",
        ]:
            if list_field in item and not isinstance(item.get(list_field), list):
                errors.append(f"Item {idx} row_id={row_id} {list_field} must be a list.")

        training_span = item.get("training_span")
        if training_span is not None and not isinstance(training_span, dict):
            errors.append(f"Item {idx} row_id={row_id} training_span must be null or object.")

    summary = {
        "schema_version": data.get("schema_version"),
        "total_items": len(items),
        "unique_row_ids": len(seen_row_ids),
        "flagged_items": sum(
            1 for item in items
            if isinstance(item, dict) and item.get("reconciliation_flags")
        ),
        "language_counts": dict(
            Counter(
                item.get("selected_language")
                for item in items
                if isinstance(item, dict)
            )
        ),
        "errors": errors,
        "warnings_count": len(warnings),
        "warnings_preview": warnings[:20],
    }

    return data, summary


def normalize_training_span(item: dict[str, Any]) -> dict[str, Any]:
    existing = item.get("training_span")
    span = deepcopy(existing) if isinstance(existing, dict) else {}

    span.setdefault("span_id", None)
    span.setdefault("language", item.get("selected_language"))
    span.setdefault("start", item.get("global_start"))
    span.setdefault("end", item.get("global_end"))
    span.setdefault("text", item.get("corrected_text"))

    return span


def normalize_review_item(item: dict[str, Any], index: int) -> dict[str, Any]:
    missing = REQUIRED_ITEM_FIELDS - set(item.keys())
    if missing:
        raise ValueError(f"Item {index} missing required fields: {sorted(missing)}")

    selected_language = item["selected_language"]
    if selected_language not in VALID_LANGUAGES:
        raise ValueError(
            f"Item {index} row_id={item.get('row_id')} "
            f"has invalid selected_language: {selected_language}"
        )

    corrected_text = item["corrected_text"]
    if not isinstance(corrected_text, str):
        raise TypeError(
            f"Item {index} row_id={item.get('row_id')} corrected_text must be a string."
        )

    return {
        "row_id": item["row_id"],
        "audio_id": item["audio_id"],
        "word_index": item["word_index"],
        "global_start": item["global_start"],
        "global_end": item["global_end"],
        "local_start": item.get("local_start"),
        "local_end": item.get("local_end"),
        "bronze_ar_text": item.get("bronze_ar_text", ""),
        "bronze_en_text": item.get("bronze_en_text", ""),
        "selected_language": selected_language,
        "corrected_text": corrected_text,
        "original_selected_language": item.get(
            "original_selected_language",
            selected_language,
        ),
        "original_corrected_text": item.get(
            "original_corrected_text",
            corrected_text,
        ),
        "language_hypotheses": item.get("language_hypotheses", []),
        "reconciliation_flags": item.get("reconciliation_flags", []),
        "left_context": item.get("left_context", []),
        "right_context": item.get("right_context", []),
        "review_status": item.get("review_status", "unreviewed"),
        "reviewed_at": item.get("reviewed_at"),
        "reviewer_id": item.get("reviewer_id"),
        "training_span": normalize_training_span(item),
        "source": {
            "schema_version": "gradio_reconciliation_input_v1",
            "source_index": index,
        },
    }


def normalize_reconciliation_input(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    source = load_json(path)

    if source.get("schema_version") != "gradio_reconciliation_input_v1":
        raise ValueError(f"Unexpected schema_version: {source.get('schema_version')}")

    items = source.get("items")
    if not isinstance(items, list):
        raise TypeError("Input field `items` must be a list.")

    normalized_items = [
        normalize_review_item(item, index)
        for index, item in enumerate(items)
    ]

    row_ids = [item["row_id"] for item in normalized_items]
    duplicates = [
        row_id
        for row_id, count in Counter(row_ids).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate row_id values found: {duplicates[:20]}")

    return {
        "schema_version": "normalized_review_items_v1",
        "created_at": utc_now_iso(),
        "source_file": str(path),
        "items": normalized_items,
        "summary": {
            "total_items": len(normalized_items),
            "flagged_items": sum(
                1 for item in normalized_items
                if item.get("reconciliation_flags")
            ),
            "language_counts": dict(
                Counter(item["selected_language"] for item in normalized_items)
            ),
        },
    }
