\
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from pipeline.common.json_io import (
    atomic_write_json,
    read_json,
)


SCHEMA_VERSION = "bronze_v2_openai_whisper_raw_v1"


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


def is_complete_output(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        payload = read_json(path)
    except Exception:
        return False

    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("inference", {}).get("status") == "completed"
        and isinstance(payload.get("result"), dict)
    )


def validate_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    segments = result.get("segments", [])

    segment_regressions: list[dict[str, Any]] = []
    word_regressions: list[dict[str, Any]] = []

    previous_segment_start: float | None = None
    previous_word_start: float | None = None
    native_word_count = 0

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
                        "segment_position": segment_position,
                        "start": start,
                        "previous_start": previous_segment_start,
                    }
                )

            previous_segment_start = start

        words = segment.get("words", []) or []
        native_word_count += len(words)

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
                        "segment_position": segment_position,
                        "word_index": word_index,
                        "word": word.get("word", ""),
                        "start": word_start,
                        "previous_start": previous_word_start,
                    }
                )

            previous_word_start = word_start

    validation = {
        "native_segment_count": len(segments),
        "native_word_count": native_word_count,
        "segment_regression_count": len(segment_regressions),
        "word_regression_count": len(word_regressions),
        "segment_regressions": segment_regressions,
        "word_regressions": word_regressions,
    }

    validation["passed"] = (
        validation["segment_regression_count"] == 0
        and validation["word_regression_count"] == 0
    )

    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run official OpenAI Whisper Large-v3 over one whole lecture."
        )
    )

    parser.add_argument("--lecture-id", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--download-root", type=Path)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--initial-prompt")
    parser.add_argument("--force", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.model != "large-v3":
        raise ValueError(
            "Bronze v2 is locked to official Whisper Large-v3."
        )

    if not args.audio.exists():
        raise FileNotFoundError(args.audio)

    if is_complete_output(args.output) and not args.force:
        print(
            {
                "lecture_id": args.lecture_id,
                "status": "skipped_existing",
                "output": str(args.output),
            }
        )
        return 0

    try:
        import whisper
    except ImportError as error:
        raise RuntimeError(
            "Install official OpenAI Whisper with: "
            "pip install openai-whisper"
        ) from error

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    started_at = time.perf_counter()

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

    transcribe_options: dict[str, Any] = {
        "task": "transcribe",
        "language": None,
        "temperature": args.temperature,
        "beam_size": args.beam_size,
        "condition_on_previous_text": True,
        "word_timestamps": True,
        "verbose": False,
        "fp16": args.device == "cuda",
    }

    if args.initial_prompt:
        transcribe_options["initial_prompt"] = args.initial_prompt

    transcription_started = time.perf_counter()

    result = model.transcribe(
        str(args.audio),
        **transcribe_options,
    )

    transcription_seconds = (
        time.perf_counter() - transcription_started
    )

    safe_result = make_json_safe(result)
    validation = validate_result(safe_result)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "lecture_id": args.lecture_id,
        "audio_filepath": str(args.audio.resolve()),
        "engine": "openai-whisper",
        "implementation": "official_openai_whisper",
        "model": args.model,
        "configuration": {
            "task": "transcribe",
            "language": None,
            "language_forced": False,
            "word_timestamps": True,
            "beam_size": args.beam_size,
            "temperature": args.temperature,
            "condition_on_previous_text": True,
            "fp16": args.device == "cuda",
            "device": args.device,
            "initial_prompt": args.initial_prompt,
        },
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "whisper_module": whisper.__file__,
        },
        "inference": {
            "status": "completed",
            "detected_language": safe_result.get("language"),
            "model_load_seconds": round(model_load_seconds, 3),
            "transcription_seconds": round(
                transcription_seconds,
                3,
            ),
            "runtime_seconds": round(
                time.perf_counter() - started_at,
                3,
            ),
            "native_segment_count": validation[
                "native_segment_count"
            ],
            "native_word_count": validation[
                "native_word_count"
            ],
        },
        "validation": validation,
        "result": safe_result,
    }

    atomic_write_json(args.output, payload)

    print(
        {
            "lecture_id": args.lecture_id,
            "status": "completed",
            "output": str(args.output),
            "engine": payload["engine"],
            "model": payload["model"],
            "detected_language": payload["inference"][
                "detected_language"
            ],
            "native_segment_count": payload["inference"][
                "native_segment_count"
            ],
            "native_word_count": payload["inference"][
                "native_word_count"
            ],
            "runtime_seconds": payload["inference"][
                "runtime_seconds"
            ],
            "validation_passed": validation["passed"],
        }
    )

    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
