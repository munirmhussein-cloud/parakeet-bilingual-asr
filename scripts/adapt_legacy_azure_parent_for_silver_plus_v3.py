from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def first_number(*values: Any) -> float | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def canonical_span(row: dict[str, Any], position: int) -> tuple[float, float, float]:
    start = first_number(row.get("segment_start"), row.get("global_start"), row.get("offset"))
    duration = first_number(row.get("duration"), row.get("segment_duration"))
    end = first_number(row.get("segment_end"), row.get("global_end"))
    if start is None:
        start = position * 20.0
    if end is None and duration is not None:
        end = start + duration
    if duration is None and end is not None:
        duration = end - start
    if end is None:
        duration = 20.0 if duration is None else duration
        end = start + duration
    if duration is None:
        duration = end - start
    return round(start, 6), round(end, 6), round(duration, 6)


def legacy_span(row: dict[str, Any], position: int) -> tuple[float, float, float]:
    start = first_number(row.get("segment_start"), row.get("global_start"), row.get("offset"))
    duration = first_number(row.get("duration"), row.get("segment_duration"))
    end = first_number(row.get("segment_end"), row.get("global_end"))
    if start is None:
        start = sum(
            first_number(previous.get("duration"), previous.get("segment_duration")) or 20.0
            for previous in []
        )
        start = position * 20.0
    if end is None and duration is not None:
        end = start + duration
    if duration is None and end is not None:
        duration = end - start
    if end is None:
        duration = 20.0 if duration is None else duration
        end = start + duration
    if duration is None:
        duration = end - start
    return float(start), float(end), float(duration)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remap legacy Azure/Pyannote parent rows to repaired Silver v3 canonical IDs by position.")
    parser.add_argument("--legacy-azure-parent", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    legacy_rows = read_jsonl(args.legacy_azure_parent)
    canonical_rows = read_jsonl(args.canonical_manifest)
    if len(legacy_rows) != len(canonical_rows):
        raise ValueError(f"Row count mismatch: legacy={len(legacy_rows)} canonical={len(canonical_rows)}")

    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    chronology_errors: list[dict[str, Any]] = []

    for position, (legacy, canonical) in enumerate(zip(legacy_rows, canonical_rows)):
        canonical_id = str(canonical["segment_id"])
        canonical_start, canonical_end, canonical_duration = canonical_span(canonical, position)
        old_start = first_number(legacy.get("segment_start"), legacy.get("global_start"))
        old_end = first_number(legacy.get("segment_end"), legacy.get("global_end"))
        old_duration = first_number(legacy.get("duration"))
        if old_start is None:
            old_start = canonical_start
        if old_end is None:
            old_end = old_start + (old_duration if old_duration is not None else canonical_duration)
        shift = canonical_start - old_start

        children = []
        previous_start = None
        for child_position, child in enumerate(legacy.get("children", []) if isinstance(legacy.get("children"), list) else []):
            if not isinstance(child, dict):
                continue
            child_copy = dict(child)
            child_start = first_number(child.get("global_start"))
            child_end = first_number(child.get("global_end"))
            if child_start is not None:
                child_start = round(child_start + shift, 6)
                child_copy["global_start"] = child_start
            if child_end is not None:
                child_end = round(child_end + shift, 6)
                child_copy["global_end"] = child_end
            child_copy["legacy_segment_id"] = child.get("segment_id")
            child_copy["parent_segment_id"] = canonical_id
            child_copy["child_position"] = child_position
            if child_start is not None and previous_start is not None and child_start < previous_start - 1e-6:
                chronology_errors.append({"segment_position": position, "child_position": child_position})
            if child_start is not None:
                previous_start = child_start
            children.append(child_copy)

        adapted = dict(legacy)
        adapted["schema_version"] = "azure_pyannote_parent_segment_v2_repaired_alignment"
        adapted["legacy_segment_id"] = legacy.get("segment_id")
        adapted["segment_id"] = canonical_id
        adapted["segment_index"] = position
        adapted["segment_position"] = position
        adapted["segment_start"] = canonical_start
        adapted["segment_end"] = canonical_end
        adapted["duration"] = canonical_duration
        adapted["audio_filepath"] = canonical.get("audio_filepath") or legacy.get("audio_filepath")
        adapted["children"] = children
        adapted["alignment"] = {
            "method": "position_locked_parent_remap",
            "legacy_parent_start": old_start,
            "legacy_parent_end": old_end,
            "canonical_parent_start": canonical_start,
            "canonical_parent_end": canonical_end,
            "child_timestamp_shift_seconds": round(shift, 6),
        }
        output_rows.append(adapted)
        audit_rows.append({
            "segment_position": position,
            "canonical_segment_id": canonical_id,
            "legacy_segment_id": legacy.get("segment_id"),
            "legacy_start": old_start,
            "legacy_end": old_end,
            "canonical_start": canonical_start,
            "canonical_end": canonical_end,
            "shift_seconds": round(shift, 6),
            "child_count": len(children),
        })

    write_jsonl(args.output, output_rows)
    report = {
        "schema_version": "silver_plus_v3_legacy_azure_alignment_report_v1",
        "legacy_parent": str(args.legacy_azure_parent),
        "canonical_manifest": str(args.canonical_manifest),
        "output": str(args.output),
        "row_count": len(output_rows),
        "segment_ids_unique": len({row["segment_id"] for row in output_rows}) == len(output_rows),
        "positions_ordered": [row["segment_position"] for row in output_rows] == list(range(len(output_rows))),
        "child_chronology_error_count": len(chronology_errors),
        "chronology_errors": chronology_errors,
        "max_abs_shift_seconds": max((abs(row["shift_seconds"]) for row in audit_rows), default=0.0),
        "passed": len(chronology_errors) == 0,
        "segments": audit_rows,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
