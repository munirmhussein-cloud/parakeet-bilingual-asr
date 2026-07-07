"""
Generate gradio_reconciliation_input_v1.json from Bronze AR and Bronze EN JSON.

This script bridges Sprint 1 Bronze outputs into Sprint 2 Gradio review input.
It expects Bronze files shaped like bronze_transcript_v1, with word-level items
containing text, timestamps, and optional language/confidence metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "gradio_reconciliation_input_v1"


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def extract_words(bronze: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract word-level records from flexible Bronze JSON.

    Supports likely locations:
    - bronze["words"]
    - bronze["items"]
    - bronze["segments"][*]["words"]
    """
    if isinstance(bronze.get("words"), list):
        return bronze["words"]

    if isinstance(bronze.get("items"), list):
        return bronze["items"]

    words: list[dict[str, Any]] = []
    for segment in bronze.get("segments", []):
        if isinstance(segment, dict):
            for word in segment.get("words", []):
                if isinstance(word, dict):
                    merged = dict(word)
                    merged.setdefault("segment_id", segment.get("segment_id"))
                    words.append(merged)

    return words


def get_word_text(word: dict[str, Any]) -> str:
    for key in ["text", "word", "token", "corrected_text"]:
        value = word.get(key)
        if isinstance(value, str):
            return value
    return ""


def get_start(word: dict[str, Any]) -> float | None:
    for key in ["global_start", "start", "start_time", "start_seconds"]:
        value = word.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def get_end(word: dict[str, Any]) -> float | None:
    for key in ["global_end", "end", "end_time", "end_seconds"]:
        value = word.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def pair_words(
    ar_words: list[dict[str, Any]],
    en_words: list[dict[str, Any]],
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    """
    Minimal deterministic pairing for Sprint 2 validation.

    Uses positional pairing. Time-aware reconciliation can replace this later
    without changing the Gradio input contract.
    """
    max_len = max(len(ar_words), len(en_words))
    pairs = []

    for index in range(max_len):
        ar_word = ar_words[index] if index < len(ar_words) else None
        en_word = en_words[index] if index < len(en_words) else None
        pairs.append((ar_word, en_word))

    return pairs


def choose_language(ar_text: str, en_text: str) -> str:
    if ar_text and not en_text:
        return "ar-AR"
    if en_text and not ar_text:
        return "en-US"
    if ar_text and en_text:
        return "ar-AR"
    return "ar-AR"


def make_flags(ar_word: dict[str, Any] | None, en_word: dict[str, Any] | None) -> list[str]:
    flags = []

    ar_text = get_word_text(ar_word or {})
    en_text = get_word_text(en_word or {})

    if ar_text and en_text and ar_text != en_text:
        flags.append("dual_bronze_text")

    ar_start = get_start(ar_word or {})
    en_start = get_start(en_word or {})

    if ar_start is not None and en_start is not None and abs(ar_start - en_start) > 0.5:
        flags.append("timestamp_mismatch")

    if not ar_text and not en_text:
        flags.append("missing_text")

    return flags


def build_context(items: list[dict[str, Any]], index: int, window: int = 2) -> tuple[list[str], list[str]]:
    left = [
        item["corrected_text"]
        for item in items[max(0, index - window):index]
        if item.get("corrected_text")
    ]
    right = [
        item["corrected_text"]
        for item in items[index + 1:index + 1 + window]
        if item.get("corrected_text")
    ]
    return left, right


def generate_gradio_items(
    bronze_ar: dict[str, Any],
    bronze_en: dict[str, Any],
    audio_id: str | None = None,
    audio_path: str | None = None,
) -> list[dict[str, Any]]:
    ar_words = extract_words(bronze_ar)
    en_words = extract_words(bronze_en)

    if audio_id is None:
        audio_id = bronze_ar.get("audio_id") or bronze_en.get("audio_id")

    if audio_id is None and audio_path:
        audio_id = Path(str(audio_path)).stem

    if audio_id is None:
        audio_id = "audio_unknown"

    items: list[dict[str, Any]] = []

    for index, (ar_word, en_word) in enumerate(pair_words(ar_words, en_words)):
        ar_word = ar_word or {}
        en_word = en_word or {}

        ar_text = get_word_text(ar_word)
        en_text = get_word_text(en_word)

        selected_language = choose_language(ar_text, en_text)
        corrected_text = ar_text if selected_language == "ar-AR" else en_text

        start = get_start(ar_word) if get_start(ar_word) is not None else get_start(en_word)
        end = get_end(ar_word) if get_end(ar_word) is not None else get_end(en_word)

        if start is None:
            start = round(index * 0.5, 2)
        if end is None:
            end = round(start + 0.4, 2)

        item = {
            "row_id": f"{audio_id}_word_{index:05d}",
            "audio_id": audio_id,
            "audio_path": audio_path,
            "word_index": index,
            "global_start": start,
            "global_end": end,
            "local_start": ar_word.get("local_start", en_word.get("local_start", start)),
            "local_end": ar_word.get("local_end", en_word.get("local_end", end)),
            "bronze_ar_text": ar_text,
            "bronze_en_text": en_text,
            "selected_language": selected_language,
            "corrected_text": corrected_text,
            "reconciliation_flags": make_flags(ar_word, en_word),
            "left_context": [],
            "right_context": [],
            "language_hypotheses": [
                {
                    "language": "ar-AR",
                    "text": ar_text,
                    "source": "bronze_ar",
                },
                {
                    "language": "en-US",
                    "text": en_text,
                    "source": "bronze_en",
                },
            ],
            "training_span": {
                "span_id": None,
                "language": selected_language,
                "start": start,
                "end": end,
                "text": corrected_text,
            },
        }

        items.append(item)

    for index, item in enumerate(items):
        left, right = build_context(items, index)
        item["left_context"] = left
        item["right_context"] = right

    return items


def generate_gradio_input(
    bronze_ar_path: str | Path,
    bronze_en_path: str | Path,
    output_path: str | Path,
    audio_id: str | None = None,
    audio_path: str | None = None,
) -> dict[str, Any]:
    bronze_ar = load_json(bronze_ar_path)
    bronze_en = load_json(bronze_en_path)

    items = generate_gradio_items(
        bronze_ar,
        bronze_en,
        audio_id=audio_id,
        audio_path=audio_path,
    )

    document = {
        "schema_version": SCHEMA_VERSION,
        "source_files": {
            "bronze_ar": str(bronze_ar_path),
            "bronze_en": str(bronze_en_path),
        },
        "source_audio": audio_path,
        "items": items,
    }

    write_json(output_path, document)
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Gradio reconciliation input from Bronze AR/EN JSON."
    )
    parser.add_argument("--bronze-ar", required=True)
    parser.add_argument("--bronze-en", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio-id", default=None)
    parser.add_argument("--audio-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = generate_gradio_input(
        bronze_ar_path=args.bronze_ar,
        bronze_en_path=args.bronze_en,
        output_path=args.output,
        audio_id=args.audio_id,
        audio_path=args.audio_path,
    )
    print(
        json.dumps(
            {
                "schema_version": document["schema_version"],
                "items": len(document["items"]),
                "flagged_items": sum(
                    1 for item in document["items"]
                    if item.get("reconciliation_flags")
                ),
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
