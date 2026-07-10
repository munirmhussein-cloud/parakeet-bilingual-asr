from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.segment_metadata import SegmentMetadataResolver


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a consolidated NeMo ASR manifest."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--segment-manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--duration-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Write the report and return success even when errors are present.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    report_path = Path(args.report)
    repo_root = Path(args.repo_root)
    resolver = SegmentMetadataResolver(args.segment_manifest)

    errors = []
    warnings = []
    valid_rows = 0
    total_duration = 0.0
    seen_audio: set[str] = set()
    seen_segments: set[str] = set()

    lines = [
        line
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for index, line in enumerate(lines, start=1):
        row_errors = []

        try:
            row = json.loads(line)
        except Exception as exc:
            errors.append(
                {
                    "row": index,
                    "code": "invalid_json",
                    "message": str(exc),
                }
            )
            continue

        audio_filepath = row.get("audio_filepath")
        duration = row.get("duration")
        text = str(row.get("text", "") or "").strip()

        if not audio_filepath:
            row_errors.append(("missing_audio_filepath", "audio_filepath is missing"))
        if not isinstance(duration, (int, float)) or duration <= 0:
            row_errors.append(("invalid_duration", f"Invalid duration: {duration!r}"))
        if not text:
            row_errors.append(("empty_transcript", "Transcript text is empty"))

        metadata = None
        if audio_filepath:
            metadata = resolver.resolve(audio_filepath=str(audio_filepath))
            if metadata is None:
                row_errors.append(
                    (
                        "unresolved_segment",
                        f"Audio path does not resolve: {audio_filepath}",
                    )
                )

        segment_id = metadata.get("segment_id") if metadata else None

        if segment_id and segment_id in seen_segments:
            row_errors.append(
                ("duplicate_segment_id", f"Duplicate segment_id: {segment_id}")
            )

        if audio_filepath and str(audio_filepath) in seen_audio:
            row_errors.append(
                ("duplicate_audio_filepath", f"Duplicate audio: {audio_filepath}")
            )

        if metadata and isinstance(duration, (int, float)):
            authoritative = float(metadata["duration"])
            if abs(float(duration) - authoritative) > args.duration_tolerance:
                row_errors.append(
                    (
                        "resolver_duration_mismatch",
                        f"Manifest={duration}, resolver={authoritative}",
                    )
                )

        if audio_filepath:
            audio_path = Path(str(audio_filepath))
            check_path = (
                audio_path
                if audio_path.is_absolute()
                else repo_root / audio_path
            )

            if not check_path.exists():
                row_errors.append(
                    ("missing_audio", f"Audio file does not exist: {audio_filepath}")
                )
            else:
                try:
                    measured = wav_duration(check_path)
                    if (
                        isinstance(duration, (int, float))
                        and abs(measured - float(duration))
                        > args.duration_tolerance
                    ):
                        row_errors.append(
                            (
                                "wav_duration_mismatch",
                                f"Manifest={duration}, WAV={measured:.3f}",
                            )
                        )
                except Exception as exc:
                    row_errors.append(("unreadable_audio", str(exc)))

        if row_errors:
            for code, message in row_errors:
                errors.append(
                    {
                        "row": index,
                        "segment_id": segment_id,
                        "audio_filepath": audio_filepath,
                        "code": code,
                        "message": message,
                    }
                )
            continue

        valid_rows += 1
        total_duration += float(duration)
        seen_audio.add(str(audio_filepath))
        if segment_id:
            seen_segments.add(str(segment_id))

    report = {
        "input": str(input_path),
        "total_rows": len(lines),
        "valid_rows": valid_rows,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "total_duration_seconds": round(total_duration, 3),
        "total_hours": round(total_duration / 3600.0, 4),
        "duration_tolerance_seconds": args.duration_tolerance,
        "errors": errors,
        "warnings": warnings,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if errors and not args.allow_errors:
        raise ValueError(
            f"Dataset validation failed with {len(errors)} errors."
        )


if __name__ == "__main__":
    main()
