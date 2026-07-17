from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


VIEW_NAMES = (
    "whole",
    "canonical_20s",
    "context_10s_stride_5s",
    "local_2p5s_contiguous",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize raw hosted Parakeet Silver v3 views "
            "to global word timestamps."
        )
    )

    parser.add_argument("--lecture-id", required=True)
    parser.add_argument("--silver-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    manifest_root = args.silver_root / "manifests"
    raw_root = args.silver_root / "raw_parakeet"
    normalized_root = args.silver_root / "normalized"

    view_reports = {}

    for view_name in VIEW_NAMES:
        manifest_path = (
            manifest_root
            / f"{args.lecture_id}_{view_name}.jsonl"
        )

        raw_dir = raw_root / view_name

        output_path = (
            normalized_root
            / f"{args.lecture_id}_{view_name}_normalized.jsonl"
        )

        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        manifest_rows = read_jsonl(manifest_path)

        normalized_rows = []
        missing_raw = []
        identity_errors = []
        chronology_errors = []

        for manifest_row in manifest_rows:
            segment_id = str(manifest_row["segment_id"])
            raw_path = raw_dir / f"{segment_id}.json"

            if not raw_path.exists():
                missing_raw.append(segment_id)
                continue

            raw = json.loads(
                raw_path.read_text(encoding="utf-8")
            )

            if str(raw.get("audio_id")) != segment_id:
                identity_errors.append(segment_id)

            window_start = float(
                manifest_row.get(
                    "global_start",
                    manifest_row.get("offset", 0.0),
                )
            )

            window_end = float(
                manifest_row.get(
                    "global_end",
                    window_start
                    + float(manifest_row["duration"]),
                )
            )

            words = []
            previous_start = None

            for position, word in enumerate(
                raw.get("words", [])
                if isinstance(raw.get("words"), list)
                else []
            ):
                if not isinstance(word, dict):
                    continue

                text = str(word.get("text", "")).strip()

                if not text:
                    continue

                local_start = float(word.get("start", 0.0))
                local_end = float(
                    word.get("end", local_start)
                )

                global_start = min(
                    max(
                        window_start + local_start,
                        window_start,
                    ),
                    window_end,
                )

                global_end = min(
                    max(
                        window_start + local_end,
                        global_start,
                    ),
                    window_end,
                )

                if (
                    previous_start is not None
                    and global_start < previous_start - 1e-6
                ):
                    chronology_errors.append(
                        {
                            "segment_id": segment_id,
                            "word_position": position,
                        }
                    )

                previous_start = global_start

                words.append(
                    {
                        "word_position": position,
                        "text": text,
                        "start": round(local_start, 6),
                        "end": round(local_end, 6),
                        "global_start":
                        round(global_start, 6),
                        "global_end":
                        round(global_end, 6),
                        "confidence":
                        word.get("confidence"),
                    }
                )

            normalized_rows.append(
                {
                    "schema_version":
                    "silver_v3_normalized_parakeet_view_v1",
                    "lecture_id":
                    args.lecture_id,
                    "view":
                    view_name,
                    "segment_id":
                    segment_id,
                    "audio_id":
                    segment_id,
                    "segment_position":
                    int(manifest_row["segment_position"]),
                    "audio_filepath":
                    manifest_row["audio_filepath"],
                    "global_start":
                    round(window_start, 6),
                    "global_end":
                    round(window_end, 6),
                    "duration":
                    round(window_end - window_start, 6),
                    "language":
                    raw.get("language"),
                    "backend":
                    raw.get("backend"),
                    "function_id":
                    raw.get("function_id"),
                    "words":
                    words,
                    "text":
                    " ".join(
                        item["text"]
                        for item in words
                    ).strip(),
                    "word_count":
                    len(words),
                    "has_text":
                    bool(words),
                }
            )

        positions = [
            row["segment_position"]
            for row in normalized_rows
        ]

        ids = [
            row["segment_id"]
            for row in normalized_rows
        ]

        passed = (
            len(normalized_rows) == len(manifest_rows)
            and not missing_raw
            and not identity_errors
            and not chronology_errors
            and positions
            == list(range(len(normalized_rows)))
            and len(ids) == len(set(ids))
        )

        write_jsonl(output_path, normalized_rows)

        view_reports[view_name] = {
            "manifest_row_count":
            len(manifest_rows),
            "normalized_row_count":
            len(normalized_rows),
            "total_word_count":
            sum(row["word_count"] for row in normalized_rows),
            "empty_document_count":
            sum(not row["has_text"] for row in normalized_rows),
            "missing_raw_count":
            len(missing_raw),
            "identity_error_count":
            len(identity_errors),
            "chronology_error_count":
            len(chronology_errors),
            "output":
            str(output_path),
            "passed":
            passed,
        }

    passed = all(
        report["passed"]
        for report in view_reports.values()
    )

    report = {
        "schema_version":
        "silver_v3_normalization_report_v1",
        "lecture_id":
        args.lecture_id,
        "wall_seconds":
        round(time.perf_counter() - started, 3),
        "views":
        view_reports,
        "passed":
        passed,
    }

    report_path = (
        args.report
        or args.silver_root
        / "silver_v3_normalization_report.json"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
