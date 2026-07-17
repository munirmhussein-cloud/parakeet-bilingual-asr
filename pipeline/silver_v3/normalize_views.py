from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path


VIEW_NAMES = (
    "whole",
    "canonical_20s",
    "context_10s_stride_5s",
    "local_2p5s_contiguous",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def finite_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def classify_raw_timing(
    raw_words: list[dict],
    *,
    window_start: float,
    window_end: float,
) -> tuple[str | None, str]:
    """Detect whether hosted timestamps are local or already global.

    The validated pilot preserved real global timestamps. Production previously
    assumed every timestamp was local and added the window offset again. This
    routine scores both coordinate systems and only interpolates when neither
    is credible.
    """
    timed: list[tuple[float, float]] = []
    for word in raw_words:
        start = finite_number(word.get("start"))
        end = finite_number(word.get("end"))
        if start is None or end is None:
            return None, "missing_fields"
        timed.append((start, end))

    if not timed:
        return None, "no_words"

    duration = max(0.0, window_end - window_start)
    unique_pairs = {(round(start, 6), round(end, 6)) for start, end in timed}
    positive_duration = sum(end - start > 1e-4 for start, end in timed)

    if len(timed) > 1 and len(unique_pairs) <= 1:
        return None, "single_timestamp"
    if len(timed) > 2 and positive_duration / len(timed) < 0.25:
        return None, "mostly_zero_duration"
    if len(timed) > 2 and all(abs(start) <= 1e-6 and abs(end) <= 1e-6 for start, end in timed):
        return None, "all_zero"

    previous = -math.inf
    for start, _ in timed:
        if start < previous - 1e-6:
            return None, "non_monotonic"
        previous = start

    tolerance = 0.75
    local_fit = sum(
        start >= -tolerance
        and end >= start - tolerance
        and end <= duration + tolerance
        for start, end in timed
    ) / len(timed)
    global_fit = sum(
        start >= window_start - tolerance
        and end >= start - tolerance
        and end <= window_end + tolerance
        for start, end in timed
    ) / len(timed)

    if global_fit >= 0.95 and global_fit > local_fit + 0.05:
        return "global", "credible_raw_global"
    if local_fit >= 0.95:
        return "local", "credible_raw_local"
    if global_fit >= 0.95:
        return "global", "credible_raw_global"
    return None, "out_of_range"


def interpolate_timing(position: int, count: int, duration: float) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 0.0
    return duration * position / count, duration * (position + 1) / count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize raw hosted Parakeet Silver v3 views to validated global word timestamps.")
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
    view_reports: dict[str, dict] = {}

    for view_name in VIEW_NAMES:
        manifest_path = manifest_root / f"{args.lecture_id}_{view_name}.jsonl"
        raw_dir = raw_root / view_name
        output_path = normalized_root / f"{args.lecture_id}_{view_name}_normalized.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        manifest_rows = read_jsonl(manifest_path)
        normalized_rows: list[dict] = []
        missing_raw: list[str] = []
        identity_errors: list[str] = []
        chronology_errors: list[dict] = []
        timing_method_counts: Counter[str] = Counter()
        timing_reason_counts: Counter[str] = Counter()
        source_word_count = 0
        normalized_word_count = 0

        for manifest_row in manifest_rows:
            segment_id = str(manifest_row["segment_id"])
            raw_path = raw_dir / f"{segment_id}.json"
            if not raw_path.exists():
                missing_raw.append(segment_id)
                continue

            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            if str(raw.get("audio_id")) != segment_id:
                identity_errors.append(segment_id)

            window_start = float(manifest_row.get("global_start", manifest_row.get("offset", 0.0)))
            window_end = float(manifest_row.get("global_end", window_start + float(manifest_row["duration"])))
            duration = max(0.0, window_end - window_start)
            raw_words = [
                word for word in (raw.get("words", []) if isinstance(raw.get("words"), list) else [])
                if isinstance(word, dict) and str(word.get("text", "")).strip()
            ]
            source_word_count += len(raw_words)
            timing_basis, timing_reason = classify_raw_timing(
                raw_words,
                window_start=window_start,
                window_end=window_end,
            )
            timing_reason_counts[timing_reason] += 1
            timing_method = {
                "global": "validated_raw_global_timestamp",
                "local": "validated_raw_local_timestamp",
                None: "monotonic_window_interpolation",
            }[timing_basis]
            timing_method_counts[timing_method] += len(raw_words)

            words: list[dict] = []
            previous_start: float | None = None
            for position, word in enumerate(raw_words):
                text = str(word.get("text", "")).strip()
                if timing_basis == "global":
                    global_start = min(max(float(word["start"]), window_start), window_end)
                    global_end = min(max(float(word["end"]), global_start), window_end)
                    local_start = global_start - window_start
                    local_end = global_end - window_start
                elif timing_basis == "local":
                    local_start = min(max(float(word["start"]), 0.0), duration)
                    local_end = min(max(float(word["end"]), local_start), duration)
                    global_start = window_start + local_start
                    global_end = window_start + local_end
                else:
                    local_start, local_end = interpolate_timing(position, len(raw_words), duration)
                    global_start = window_start + local_start
                    global_end = window_start + local_end

                if previous_start is not None and global_start < previous_start - 1e-6:
                    chronology_errors.append({"segment_id": segment_id, "word_position": position})
                previous_start = global_start
                words.append({
                    "word_position": position,
                    "text": text,
                    "start": round(local_start, 6),
                    "end": round(local_end, 6),
                    "global_start": round(global_start, 6),
                    "global_end": round(global_end, 6),
                    "confidence": word.get("confidence"),
                    "timing_method": timing_method,
                    "raw_timing_reason": timing_reason,
                })

            normalized_word_count += len(words)
            normalized_rows.append({
                "schema_version": "silver_v3_normalized_parakeet_view_v3",
                "lecture_id": args.lecture_id,
                "view": view_name,
                "segment_id": segment_id,
                "audio_id": segment_id,
                "segment_position": int(manifest_row["segment_position"]),
                "audio_filepath": manifest_row["audio_filepath"],
                "global_start": round(window_start, 6),
                "global_end": round(window_end, 6),
                "duration": round(duration, 6),
                "language": raw.get("language"),
                "backend": raw.get("backend"),
                "function_id": raw.get("function_id"),
                "timing_method": timing_method,
                "raw_timing_reason": timing_reason,
                "words": words,
                "text": " ".join(item["text"] for item in words).strip(),
                "word_count": len(words),
                "has_text": bool(words),
            })

        positions = [row["segment_position"] for row in normalized_rows]
        ids = [row["segment_id"] for row in normalized_rows]
        source_accounting_closed = source_word_count == normalized_word_count
        passed = (
            len(normalized_rows) == len(manifest_rows)
            and not missing_raw
            and not identity_errors
            and not chronology_errors
            and positions == list(range(len(normalized_rows)))
            and len(ids) == len(set(ids))
            and source_accounting_closed
        )
        write_jsonl(output_path, normalized_rows)
        view_reports[view_name] = {
            "manifest_row_count": len(manifest_rows),
            "normalized_row_count": len(normalized_rows),
            "source_word_count": source_word_count,
            "normalized_word_count": normalized_word_count,
            "source_accounting_closed": source_accounting_closed,
            "total_word_count": normalized_word_count,
            "empty_document_count": sum(not row["has_text"] for row in normalized_rows),
            "missing_raw_count": len(missing_raw),
            "identity_error_count": len(identity_errors),
            "chronology_error_count": len(chronology_errors),
            "timing_method_counts": dict(timing_method_counts),
            "timing_reason_document_counts": dict(timing_reason_counts),
            "output": str(output_path),
            "passed": passed,
        }

    passed = all(report["passed"] for report in view_reports.values())
    report = {
        "schema_version": "silver_v3_normalization_report_v3",
        "lecture_id": args.lecture_id,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "views": view_reports,
        "passed": passed,
    }
    report_path = args.report or args.silver_root / "silver_v3_normalization_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
