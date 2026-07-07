"""
Riva endpoint inference helpers.

Produces Bronze-compatible word-level ASR JSON from a WAV file using the
NVIDIA Riva gRPC endpoint discovered in Sprint 1.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_RIVA_SERVER = "grpc.nvcf.nvidia.com:443"
DEFAULT_RIVA_FUNCTION_ID = "71203149-d3b7-4460-8231-1be2543a1fca"


def _word_seconds(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    seconds = getattr(value, "seconds", None)
    nanos = getattr(value, "nanos", None)

    if seconds is not None or nanos is not None:
        return float(seconds or 0) + float(nanos or 0) / 1_000_000_000

    return None


def _extract_words_from_riva_response(response: Any) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []

    for result in getattr(response, "results", []):
        alternatives = getattr(result, "alternatives", [])
        if not alternatives:
            continue

        alternative = alternatives[0]

        for index, word_info in enumerate(getattr(alternative, "words", [])):
            text = getattr(word_info, "word", "")
            start = _word_seconds(getattr(word_info, "start_time", None))
            end = _word_seconds(getattr(word_info, "end_time", None))
            confidence = getattr(word_info, "confidence", None)

            if start is None:
                start = round(len(words) * 0.5, 2)
            if end is None:
                end = round(float(start) + 0.4, 2)

            words.append(
                {
                    "text": text,
                    "start": float(start),
                    "end": float(end),
                    "confidence": confidence,
                }
            )

    if words:
        return words

    # Fallback if endpoint returns transcript text but no word offsets.
    transcript_parts: list[str] = []
    for result in getattr(response, "results", []):
        alternatives = getattr(result, "alternatives", [])
        if alternatives:
            transcript = getattr(alternatives[0], "transcript", "")
            if transcript:
                transcript_parts.append(transcript)

    transcript = " ".join(transcript_parts).strip()

    return [
        {
            "text": token,
            "start": round(index * 0.5, 2),
            "end": round(index * 0.5 + 0.4, 2),
            "confidence": None,
        }
        for index, token in enumerate(transcript.split())
    ]


def transcribe_file(
    audio_path: str | Path,
    *,
    model_name: str | None = None,
    language: str,
    audio_id: str | None = None,
    server: str = DEFAULT_RIVA_SERVER,
    function_id: str = DEFAULT_RIVA_FUNCTION_ID,
    api_key: str | None = None,
    use_ssl: bool = True,
    automatic_punctuation: bool = True,
    verbatim_transcripts: bool = True,
    max_alternatives: int = 1,
) -> dict[str, Any]:
    """
    Transcribe one WAV file through Riva and return Bronze-compatible JSON.

    Args:
        model_name: Optional Riva model name. For the hosted Sprint 1 endpoint,
            language forcing is controlled primarily through language_code.
        language: Forced language code, e.g. ar-AR or en-US.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    api_key = api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing NVIDIA API key. Set NVIDIA_API_KEY or pass --api-key-env."
        )

    try:
        import riva.client
    except Exception as exc:
        raise RuntimeError(
            "nvidia-riva-client is not installed. Run: pip install -U nvidia-riva-client"
        ) from exc

    metadata = [
        ("function-id", function_id),
        ("authorization", f"Bearer {api_key}"),
    ]

    auth = riva.client.Auth(
        uri=server,
        use_ssl=use_ssl,
        metadata_args=metadata,
    )

    asr_service = riva.client.ASRService(auth)

    config_kwargs = {
        "language_code": language,
        "max_alternatives": max_alternatives,
        "enable_word_time_offsets": True,
        "enable_automatic_punctuation": automatic_punctuation,
        "verbatim_transcripts": verbatim_transcripts,
    }

    if model_name:
        config_kwargs["model"] = model_name

    config = riva.client.RecognitionConfig(**config_kwargs)

    with audio_path.open("rb") as handle:
        audio_bytes = handle.read()

    response = asr_service.offline_recognize(audio_bytes, config)
    words = _extract_words_from_riva_response(response)

    return {
        "schema_version": "bronze_transcript_v1",
        "audio_id": audio_id or audio_path.stem,
        "audio_path": str(audio_path),
        "backend": "riva_endpoint",
        "server": server,
        "function_id": function_id,
        "model_name": model_name,
        "language": language,
        "words": words,
    }
