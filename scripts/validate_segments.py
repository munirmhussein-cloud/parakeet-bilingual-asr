#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path


def ffprobe_duration(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(metadata_path, report_out, min_duration=1.0, max_duration=30.0, tolerance=0.25):
    rows = read_jsonl(metadata_path)
    seen = set()
    errors = []
    warnings = []

    for row in rows:
        segment_id = row.get("segment_id")
        segment_path = Path(row.get("segment_audio_filepath", ""))

        if segment_id in seen:
            errors.append({"segment_id": segment_id, "error": "duplicate_segment_id"})
        seen.add(segment_id)

        if not segment_path.exists():
            errors.append({"segment_id": segment_id, "error": "missing_audio_file"})
            continue

        if segment_path.stat().st_size == 0:
            errors.append({"segment_id": segment_id, "error": "empty_audio_file"})
            continue

        start = float(row.get("start_time", -1))
        end = float(row.get("end_time", -1))
        duration = float(row.get("duration", -1))

        if start < 0 or end <= start or duration <= 0:
            errors.append({"segment_id": segment_id, "error": "invalid_timing"})

        if duration < min_duration:
            warnings.append({"segment_id": segment_id, "warning": "short_segment", "duration": duration})

        if duration > max_duration:
            warnings.append({"segment_id": segment_id, "warning": "long_segment", "duration": duration})

        actual_duration = ffprobe_duration(segment_path)
        if abs(actual_duration - duration) > tolerance:
            errors.append({
                "segment_id": segment_id,
                "error": "duration_mismatch",
                "metadata_duration": duration,
                "actual_duration": round(actual_duration, 3)
            })

    report = {
        "metadata_path": str(metadata_path),
        "segment_count": len(rows),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "validation_status": "passed" if not errors else "failed"
    }

    report_out = Path(report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    with report_out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if errors:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    args = parser.parse_args()

    validate(args.metadata, args.report_out, args.min_duration, args.max_duration)


if __name__ == "__main__":
    main()
