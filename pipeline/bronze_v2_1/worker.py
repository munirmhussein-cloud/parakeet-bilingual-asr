\
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from pipeline.common.json_io import (
    atomic_write_json,
    read_json,
    read_jsonl,
)


SCHEMA_VERSION = (
    "bronze_v2_1_openai_whisper_segment_raw_v1"
)


def make_json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def is_complete(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        payload = read_json(path)
    except Exception:
        return False

    return (
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("inference", {}).get("status")
        == "completed"
        and isinstance(payload.get("result"), dict)
    )


def resolve_segment_id(
    row: dict[str, Any],
    index: int,
) -> str:
    return str(
        row.get("segment_id")
        or row.get("audio_id")
        or f"segment_{index:06d}"
    )


def resolve_audio_path(
    row: dict[str, Any],
) -> Path:
    value = (
        row.get("audio_filepath")
        or row.get("audio_path")
    )

    if not value:
        raise ValueError(
            "Manifest row has no audio filepath."
        )

    return Path(str(value))


def resolve_number(
    row: dict[str, Any],
    keys: list[str],
) -> float | None:
    for key in keys:
        value = row.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return None


def resolve_span(
    row: dict[str, Any],
    index: int,
) -> tuple[float, float, float]:
    start = resolve_number(
        row,
        [
            "segment_start",
            "start",
            "offset",
            "global_start",
        ],
    )

    end = resolve_number(
        row,
        [
            "segment_end",
            "end",
            "global_end",
        ],
    )

    duration = resolve_number(
        row,
        [
            "duration",
            "segment_duration",
        ],
    )

    if start is None:
        start = index * 20.0

    if end is None and duration is not None:
        end = start + duration

    if duration is None and end is not None:
        duration = end - start

    if end is None:
        duration = 20.0 if duration is None else duration
        end = start + duration

    if duration is None:
        duration = end - start

    return (
        round(start, 6),
        round(end, 6),
        round(duration, 6),
    )


def validate_local_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    segments = result.get("segments", []) or []

    segment_regressions = []
    word_regressions = []

    previous_segment_start = None
    previous_word_start = None
    word_count = 0

    for segment_position, segment in enumerate(segments):
        start = segment.get("start")

        if start is not None:
            start = float(start)

            if (
                previous_segment_start is not None
                and start < previous_segment_start
            ):
                segment_regressions.append(
                    {
                        "segment_position":
                        segment_position,
                        "start": start,
                        "previous_start":
                        previous_segment_start,
                    }
                )

            previous_segment_start = start

        words = segment.get("words", []) or []
        word_count += len(words)

        for word_index, word in enumerate(words):
            word_start = word.get("start")

            if word_start is None:
                continue

            word_start = float(word_start)

            if (
                previous_word_start is not None
                and word_start < previous_word_start
            ):
                word_regressions.append(
                    {
                        "segment_position":
                        segment_position,
                        "word_index": word_index,
                        "word": word.get("word", ""),
                        "start": word_start,
                        "previous_start":
                        previous_word_start,
                    }
                )

            previous_word_start = word_start

    return {
        "native_segment_count": len(segments),
        "native_word_count": word_count,
        "segment_regression_count":
        len(segment_regressions),
        "word_regression_count":
        len(word_regressions),
        "segment_regressions":
        segment_regressions,
        "word_regressions":
        word_regressions,
        "passed": (
            len(segment_regressions) == 0
            and len(word_regressions) == 0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lecture-id",
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--runner-id",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--runner-count",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--model",
        default="large-v3",
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.model != "large-v3":
        raise ValueError(
            "Bronze v2.1 is locked to large-v3."
        )

    rows = read_jsonl(args.manifest)

    assigned = [
        (index, row)
        for index, row in enumerate(rows)
        if index % args.runner_count
        == args.runner_id
    ]

    pending = []

    for index, row in assigned:
        segment_id = resolve_segment_id(
            row,
            index,
        )

        output_path = (
            args.output_dir
            / f"{segment_id}.json"
        )

        if (
            not args.force
            and is_complete(output_path)
        ):
            continue

        pending.append(
            (
                index,
                segment_id,
                row,
                output_path,
            )
        )

    if not pending:
        print(
            {
                "runner_id": args.runner_id,
                "assigned": len(assigned),
                "status": "nothing_pending",
            }
        )
        return 0

    import whisper

    runner_started = time.perf_counter()

    load_started = time.perf_counter()

    model = whisper.load_model(
        args.model,
        device=args.device,
        download_root=(
            str(args.download_root)
            if args.download_root
            else None
        ),
    )

    model_load_seconds = (
        time.perf_counter() - load_started
    )

    completed = 0
    failures = []
    total_transcription_seconds = 0.0

    for (
        index,
        segment_id,
        row,
        output_path,
    ) in pending:
        try:
            audio_path = resolve_audio_path(row)

            if not audio_path.exists():
                raise FileNotFoundError(audio_path)

            start, end, duration = resolve_span(
                row,
                index,
            )

            transcribe_started = (
                time.perf_counter()
            )

            result = model.transcribe(
                str(audio_path),
                task="transcribe",
                language=None,
                temperature=args.temperature,
                beam_size=args.beam_size,
                condition_on_previous_text=False,
                word_timestamps=True,
                verbose=False,
                fp16=args.device == "cuda",
            )

            transcription_seconds = (
                time.perf_counter()
                - transcribe_started
            )

            total_transcription_seconds += (
                transcription_seconds
            )

            safe_result = make_json_safe(result)

            validation = (
                validate_local_result(
                    safe_result
                )
            )

            payload = {
                "schema_version":
                SCHEMA_VERSION,
                "lecture_id":
                args.lecture_id,
                "segment_index":
                index,
                "segment_id":
                segment_id,
                "audio_filepath":
                str(audio_path),
                "canonical": {
                    "segment_start": start,
                    "segment_end": end,
                    "duration": duration,
                },
                "runner": {
                    "runner_id":
                    args.runner_id,
                    "runner_count":
                    args.runner_count,
                    "assignment_method":
                    "segment_index_modulo_runner_count",
                },
                "engine":
                "openai-whisper",
                "implementation":
                "official_openai_whisper",
                "model":
                args.model,
                "configuration": {
                    "language": None,
                    "language_forced": False,
                    "beam_size":
                    args.beam_size,
                    "temperature":
                    args.temperature,
                    "condition_on_previous_text":
                    False,
                    "word_timestamps": True,
                },
                "inference": {
                    "status": "completed",
                    "detected_language":
                    safe_result.get("language"),
                    "model_load_seconds_for_runner":
                    round(
                        model_load_seconds,
                        3,
                    ),
                    "transcription_seconds":
                    round(
                        transcription_seconds,
                        3,
                    ),
                },
                "validation":
                validation,
                "result":
                safe_result,
            }

            atomic_write_json(
                output_path,
                payload,
            )

            completed += 1

        except Exception as error:
            failures.append(
                {
                    "segment_index": index,
                    "segment_id": segment_id,
                    "error": repr(error),
                }
            )

    summary = {
        "schema_version":
        "bronze_v2_1_worker_summary_v1",
        "lecture_id":
        args.lecture_id,
        "runner_id":
        args.runner_id,
        "runner_count":
        args.runner_count,
        "assigned":
        len(assigned),
        "attempted":
        len(pending),
        "completed":
        completed,
        "skipped_existing":
        len(assigned) - len(pending),
        "failed":
        len(failures),
        "model_load_seconds":
        round(model_load_seconds, 3),
        "total_transcription_seconds":
        round(
            total_transcription_seconds,
            3,
        ),
        "runner_wall_seconds":
        round(
            time.perf_counter()
            - runner_started,
            3,
        ),
        "failures":
        failures,
    }

    summary_path = (
        args.output_dir
        / (
            f"_runner_"
            f"{args.runner_id:02d}_summary.json"
        )
    )

    atomic_write_json(
        summary_path,
        summary,
    )

    print(summary)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
