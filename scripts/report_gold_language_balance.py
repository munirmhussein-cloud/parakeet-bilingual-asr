from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.gold_dataset import (
    corrected_text,
    discover_gold_files,
    evaluate_gold_document,
    load_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report language balance directly from selected Gold files."
    )
    parser.add_argument("--gold-dir", default=None)
    parser.add_argument("--gold-file", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
    )
    args = parser.parse_args()

    gold_files = discover_gold_files(args.gold_dir, args.gold_file)
    if not gold_files:
        raise ValueError("No Gold files were selected.")

    row_counts = Counter()
    character_counts = Counter()
    segment_presence = Counter()
    span_ms = defaultdict(float)
    mixed_segments = []
    excluded_files = []

    total_segments = 0
    total_nonempty_rows = 0

    for gold_path in gold_files:
        try:
            document, evaluation = evaluate_gold_document(gold_path)
        except Exception as exc:
            excluded_files.append(
                {"gold_file": str(gold_path), "error": str(exc)}
            )
            continue

        items = document.get("items", [])
        languages_in_segment = set()

        for item in items:
            text = corrected_text(item)
            if not text:
                continue

            if (
                not args.include_unreviewed
                and item.get("review_status") != "reviewed"
            ):
                continue

            language = str(item.get("selected_language") or "unknown")
            row_counts[language] += 1
            character_counts[language] += len(text)
            languages_in_segment.add(language)
            total_nonempty_rows += 1

            start = item.get("local_start")
            end = item.get("local_end")
            if (
                isinstance(start, (int, float))
                and isinstance(end, (int, float))
                and end >= start
            ):
                span_ms[language] += float(end) - float(start)

        if languages_in_segment:
            total_segments += 1
            for language in languages_in_segment:
                segment_presence[language] += 1

            if len(languages_in_segment) > 1:
                mixed_segments.append(
                    {
                        "gold_file": str(gold_path),
                        "segment_id": evaluation.get("segment_id"),
                        "languages": sorted(languages_in_segment),
                    }
                )

    total_rows = sum(row_counts.values())
    total_characters = sum(character_counts.values())

    languages = sorted(
        set(row_counts)
        | set(character_counts)
        | set(segment_presence)
        | set(span_ms)
    )

    report = {
        "gold_files_selected": len(gold_files),
        "segments_with_reviewed_text": total_segments,
        "nonempty_annotation_rows": total_nonempty_rows,
        "mixed_language_segments": len(mixed_segments),
        "languages": {
            language: {
                "annotation_rows": row_counts[language],
                "row_percentage": (
                    round(row_counts[language] / total_rows * 100.0, 2)
                    if total_rows
                    else 0.0
                ),
                "characters": character_counts[language],
                "character_percentage": (
                    round(
                        character_counts[language]
                        / total_characters
                        * 100.0,
                        2,
                    )
                    if total_characters
                    else 0.0
                ),
                "segments_present": segment_presence[language],
                "reviewed_span_seconds": round(
                    span_ms[language] / 1000.0,
                    3,
                ),
            }
            for language in languages
        },
        "mixed_segment_details": mixed_segments,
        "excluded_files": excluded_files,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
