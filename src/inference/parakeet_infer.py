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


class RivaTranscriber:
    """Reusable Riva ASR client for multiple segment requests."""

    def __init__(
        self,
        *,
        language: str,
        model_name: str | None = None,
        server: str = DEFAULT_RIVA_SERVER,
        function_id: str = DEFAULT_RIVA_FUNCTION_ID,
        api_key: str | None = None,
        use_ssl: bool = True,
        automatic_punctuation: bool = True,
        verbatim_transcripts: bool = True,
        max_alternatives: int = 1,
    ) -> None:
        api_key = api_key or os.environ.get("NVIDIA_API_KEY")

        if not api_key:
            raise RuntimeError(
                "Missing NVIDIA API key. Set NVIDIA_API_KEY."
            )

        try:
            import riva.client
        except Exception as exc:
            raise RuntimeError(
                "nvidia-riva-client is not installed."
            ) from exc

        metadata = [
            ("function-id", function_id),
            ("authorization", f"Bearer {api_key}"),
        ]

        self.auth = riva.client.Auth(
            uri=server,
            use_ssl=use_ssl,
            metadata_args=metadata,
        )
        self.asr_service = riva.client.ASRService(self.auth)

        config_kwargs = {
            "language_code": language,
            "max_alternatives": max_alternatives,
            "enable_word_time_offsets": True,
            "enable_automatic_punctuation": automatic_punctuation,
            "verbatim_transcripts": verbatim_transcripts,
        }

        if model_name:
            config_kwargs["model"] = model_name

        self.config = riva.client.RecognitionConfig(
            **config_kwargs
        )

        self.language = language
        self.model_name = model_name
        self.server = server
        self.function_id = function_id

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        audio_id: str | None = None,
    ) -> dict[str, Any]:
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        with audio_path.open("rb") as handle:
            audio_bytes = handle.read()

        response = self.asr_service.offline_recognize(
            audio_bytes,
            self.config,
        )

        words = _extract_words_from_riva_response(response)

        return {
            "schema_version": "bronze_transcript_v1",
            "audio_id": audio_id or audio_path.stem,
            "audio_path": str(audio_path),
            "backend": "riva_endpoint",
            "server": self.server,
            "function_id": self.function_id,
            "model_name": self.model_name,
            "language": self.language,
            "words": words,
        }


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
    """Backward-compatible single-file inference wrapper."""
    client = RivaTranscriber(
        language=language,
        model_name=model_name,
        server=server,
        function_id=function_id,
        api_key=api_key,
        use_ssl=use_ssl,
        automatic_punctuation=automatic_punctuation,
        verbatim_transcripts=verbatim_transcripts,
        max_alternatives=max_alternatives,
    )

    return client.transcribe(
        audio_path,
        audio_id=audio_id,
    )

