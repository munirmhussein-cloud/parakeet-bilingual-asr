#!/usr/bin/env python3
"""Generate a Bronze v2.3 whole-file transcript with Faster-Whisper Turbo.

Designed for Google Colab with an NVIDIA A100. The runner transcribes one complete
lecture audio file, preserving Whisper's sequential context while exporting both
segment- and word-level timestamps in a deterministic JSON package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "bronze_v2_3_faster_whisper_whole_v1"
PIPELINE_VERSION = "bronze_v2.3"
DEFAULT_MODEL = "turbo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lecture-id", required=True, help="Canonical ID, e.g. lecture_001")
    parser.add_argument("--audio", required=True, type=Path, help="Source audio path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Faster-Whisper model name or local path")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--compute-type", default="float16", help="CTranslate2 compute type")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--language", default=None, help="Force ISO language code; default auto-detect")
    parser.add_argument("--task", choices=("transcribe", "translate"), default="transcribe")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--download-root", type=Path, default=None)
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument("--vad-filter", action="store_true", help="Enable Silero VAD; disabled by default for whole-file continuity")
    parser.add_argument("--no-word-timestamps", action="store_true")
    parser.add_argument("--no-condition-on-previous-text", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return None
    value = (result.stdout or result.stderr).strip()
    return value or None


def gpu_metadata() -> dict[str, Any]:
    query = command_output([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not query:
        return {"available": False}
    first = query.splitlines()[0]
    fields = [item.strip() for item in first.split(",")]
    return {
        "available": True,
        "name": fields[0] if len(fields) > 0 else None,
        "memory_total_mib": int(fields[1]) if len(fields) > 1 and fields[1].isdigit() else None,
        "driver_version": fields[2] if len(fields) > 2 else None,
    }


def plain_object(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "_asdict"):
        return dict(value._asdict())
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def serialize_word(word: Any) -> dict[str, Any]:
    raw = plain_object(word)
    return {
        "start": float(raw.get("start", getattr(word, "start", 0.0))),
        "end": float(raw.get("end", getattr(word, "end", 0.0))),
        "word": str(raw.get("word", getattr(word, "word", ""))),
        "probability": float(raw.get("probability", getattr(word, "probability", 0.0))),
    }


def serialize_segments(segments: Iterable[Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for segment in segments:
        raw = plain_object(segment)
        words = raw.get("words", getattr(segment, "words", None)) or []
        output.append({
            "id": int(raw.get("id", getattr(segment, "id", len(output)))),
            "seek": int(raw.get("seek", getattr(segment, "seek", 0))),
            "start": float(raw.get("start", getattr(segment, "start", 0.0))),
            "end": float(raw.get("end", getattr(segment, "end", 0.0))),
            "text": str(raw.get("text", getattr(segment, "text", ""))).strip(),
            "tokens": list(raw.get("tokens", getattr(segment, "tokens", [])) or []),
            "temperature": float(raw.get("temperature", getattr(segment, "temperature", 0.0))),
            "avg_logprob": float(raw.get("avg_logprob", getattr(segment, "avg_logprob", 0.0))),
            "compression_ratio": float(raw.get("compression_ratio", getattr(segment, "compression_ratio", 0.0))),
            "no_speech_prob": float(raw.get("no_speech_prob", getattr(segment, "no_speech_prob", 0.0))),
            "words": [serialize_word(word) for word in words],
        })
    return output


def validate_result(payload: dict[str, Any], lecture_id: str, audio: Path) -> None:
    if payload.get("lecture_id") != lecture_id:
        raise RuntimeError("Output lecture identity validation failed")
    source = payload.get("source_audio", {})
    if source.get("path") != str(audio.resolve()):
        raise RuntimeError("Output source audio validation failed")
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("Transcription produced no segments")
    transcript = payload.get("text", "").strip()
    if not transcript:
        raise RuntimeError("Transcription produced empty text")
    previous_end = -1.0
    for index, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        if start < 0 or end < start:
            raise RuntimeError(f"Invalid timestamps in segment {index}")
        if start + 0.01 < previous_end:
            raise RuntimeError(f"Non-monotonic segment timestamps at segment {index}")
        previous_end = end


def main() -> int:
    args = parse_args()
    audio = args.audio.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not audio.is_file():
        raise FileNotFoundError(f"Audio does not exist: {audio}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
    if args.device == "cuda" and not gpu_metadata().get("available"):
        raise RuntimeError("CUDA was requested but nvidia-smi did not detect a GPU")

    try:
        import ctranslate2
        import faster_whisper
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing Faster-Whisper dependencies. Install with: pip install faster-whisper"
        ) from exc

    model_kwargs: dict[str, Any] = {
        "device": args.device,
        "compute_type": args.compute_type,
        "cpu_threads": args.cpu_threads,
        "num_workers": args.num_workers,
    }
    if args.download_root is not None:
        args.download_root.mkdir(parents=True, exist_ok=True)
        model_kwargs["download_root"] = str(args.download_root.resolve())

    started_at = utc_now()
    started_clock = time.perf_counter()
    model = WhisperModel(args.model, **model_kwargs)

    transcribe_kwargs: dict[str, Any] = {
        "task": args.task,
        "language": args.language,
        "beam_size": args.beam_size,
        "best_of": args.best_of,
        "temperature": args.temperature,
        "word_timestamps": not args.no_word_timestamps,
        "condition_on_previous_text": not args.no_condition_on_previous_text,
        "vad_filter": args.vad_filter,
        "initial_prompt": args.initial_prompt,
        "multilingual": True,
        "log_progress": True,
    }

    segment_iterator, info = model.transcribe(str(audio), **transcribe_kwargs)
    segments = serialize_segments(segment_iterator)
    elapsed_seconds = time.perf_counter() - started_clock
    info_raw = plain_object(info)
    transcript_text = " ".join(segment["text"] for segment in segments if segment["text"]).strip()
    audio_duration = float(info_raw.get("duration", getattr(info, "duration", 0.0)) or 0.0)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "lecture_id": args.lecture_id,
        "created_at": utc_now(),
        "source_audio": {
            "path": str(audio),
            "filename": audio.name,
            "size_bytes": audio.stat().st_size,
            "sha256": sha256_file(audio),
        },
        "engine": {
            "name": "faster-whisper",
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "faster_whisper_version": getattr(faster_whisper, "__version__", None),
            "ctranslate2_version": getattr(ctranslate2, "__version__", None),
        },
        "runtime": {
            "started_at": started_at,
            "completed_at": utc_now(),
            "elapsed_seconds": elapsed_seconds,
            "audio_duration_seconds": audio_duration,
            "real_time_factor": elapsed_seconds / audio_duration if audio_duration > 0 else None,
            "python": sys.version,
            "platform": platform.platform(),
            "gpu": gpu_metadata(),
        },
        "transcription_options": transcribe_kwargs,
        "language": info_raw.get("language", getattr(info, "language", None)),
        "language_probability": info_raw.get(
            "language_probability", getattr(info, "language_probability", None)
        ),
        "duration": audio_duration,
        "duration_after_vad": info_raw.get(
            "duration_after_vad", getattr(info, "duration_after_vad", None)
        ),
        "text": transcript_text,
        "segment_count": len(segments),
        "word_count": len(transcript_text.split()),
        "segments": segments,
    }

    validate_result(payload, args.lecture_id, audio)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)

    print("\n=== BRONZE V2.3 COMPLETE ===")
    print(f"Lecture: {args.lecture_id}")
    print(f"Audio: {audio}")
    print(f"Output: {output}")
    print(f"Segments: {len(segments):,}")
    print(f"Words: {payload['word_count']:,}")
    print(f"Detected language: {payload['language']}")
    print(f"Elapsed: {elapsed_seconds:,.1f} seconds")
    if payload["runtime"]["real_time_factor"] is not None:
        print(f"Real-time factor: {payload['runtime']['real_time_factor']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
