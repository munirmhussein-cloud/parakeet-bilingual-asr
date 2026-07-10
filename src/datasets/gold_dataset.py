from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


PUNCTUATION = [".", ",", "?", "!", ":", ";"]


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise TypeError(f"Gold document must be an object: {path}")

    return data


def discover_gold_files(
    gold_dir: str | Path | None = None,
    gold_files: Iterable[str | Path] | None = None,
) -> list[Path]:
    paths: set[Path] = set()

    if gold_dir:
        directory = Path(gold_dir)
        if not directory.exists():
            raise FileNotFoundError(f"Gold directory does not exist: {directory}")

        # Use JSON documents only; JSONL contains duplicate representations.
        paths.update(directory.glob("*.json"))

    for value in gold_files or []:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Gold file does not exist: {path}")
        paths.add(path)

    return sorted(paths)


def corrected_text(item: dict[str, Any]) -> str:
    return str(item.get("corrected_text", "") or "").strip()


def nonempty_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if corrected_text(item)]


def clean_join(tokens: Iterable[str]) -> str:
    text = " ".join(str(token).strip() for token in tokens if str(token).strip())

    for punctuation in PUNCTUATION:
        text = text.replace(f" {punctuation}", punctuation)

    return text.strip()


def get_segment_id(document: dict[str, Any], items: list[dict[str, Any]]) -> str | None:
    candidates = [
        document.get("segment_id"),
        document.get("audio_id"),
    ]

    for item in items:
        candidates.extend([item.get("segment_id"), item.get("audio_id")])

    for candidate in candidates:
        if candidate:
            return str(candidate)

    return None


def evaluate_gold_document(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path)
    document = load_json(path)
    items = document.get("items", [])

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if not isinstance(items, list):
        errors.append(
            {
                "code": "invalid_items",
                "message": "Gold field `items` must be a list.",
            }
        )
        items = []

    populated = nonempty_items(items)
    unreviewed = [
        item
        for item in populated
        if item.get("review_status") != "reviewed"
    ]

    if not populated:
        errors.append(
            {
                "code": "empty_gold_text",
                "message": "No non-empty corrected annotation rows were found.",
            }
        )

    if unreviewed:
        errors.append(
            {
                "code": "incomplete_review",
                "message": (
                    f"{len(unreviewed)} non-empty annotation rows are not reviewed."
                ),
                "row_ids": [item.get("row_id") for item in unreviewed[:20]],
            }
        )

    segment_id = get_segment_id(document, items)
    if not segment_id:
        errors.append(
            {
                "code": "missing_segment_id",
                "message": "No audio_id or segment_id was found.",
            }
        )

    text = clean_join(corrected_text(item) for item in populated)
    if not text:
        errors.append(
            {
                "code": "empty_joined_text",
                "message": "Joined Gold transcript is empty.",
            }
        )

    return document, {
        "gold_file": str(path),
        "segment_id": segment_id,
        "total_rows": len(items),
        "nonempty_rows": len(populated),
        "reviewed_nonempty_rows": len(populated) - len(unreviewed),
        "text": text,
        "eligible": not errors,
        "errors": errors,
        "warnings": warnings,
    }
