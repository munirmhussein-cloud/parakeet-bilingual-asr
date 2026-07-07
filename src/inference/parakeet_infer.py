"""
Parakeet / NeMo inference helpers.

Produces Bronze-compatible word-level ASR JSON from a WAV file when NeMo ASR
models are available in the Colab runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_nemo_asr_model(model_name: str):
    try:
        from nemo.collections.asr.models import ASRModel
    except Exception as exc:
        raise RuntimeError(
            "NeMo ASR is not available. Install NeMo in Colab before running "
            "live Bronze inference."
        ) from exc

    return ASRModel.from_pretrained(model_name=model_name)


def _normalize_word(word: Any, fallback_index: int) -> dict[str, Any]:
    if isinstance(word, dict):
        text = word.get("word") or word.get("text") or word.get("token") or ""
        start = word.get("start_time", word.get("start", word.get("global_start")))
        end = word.get("end_time", word.get("end", word.get("global_end")))
        confidence = word.get("confidence")
    else:
        text = getattr(word, "word", None) or getattr(word, "text", None) or str(word)
        start = getattr(word, "start_time", None) or getattr(word, "start", None)
        end = getattr(word, "end_time", None) or getattr(word, "end", None)
        confidence = getattr(word, "confidence", None)

    if start is None:
        start = round(fallback_index * 0.5, 2)
    if end is None:
        end = round(float(start) + 0.4, 2)

    return {
        "text": str(text),
        "start": float(start),
        "end": float(end),
        "confidence": confidence,
    }


def _extract_words_from_hypothesis(hypothesis: Any) -> list[dict[str, Any]]:
    words = None

    if isinstance(hypothesis, dict):
        words = hypothesis.get("words") or hypothesis.get("word_timestamps")
        text = hypothesis.get("text", "")
    else:
        words = getattr(hypothesis, "words", None) or getattr(
            hypothesis, "word_timestamps", None
        )
        text = getattr(hypothesis, "text", str(hypothesis))

    if words:
        return [_normalize_word(word, index) for index, word in enumerate(words)]

    # Fallback when model returns only transcript text.
    return [
        {
            "text": token,
            "start": round(index * 0.5, 2),
            "end": round(index * 0.5 + 0.4, 2),
            "confidence": None,
        }
        for index, token in enumerate(str(text).split())
    ]


def transcribe_file(
    audio_path: str | Path,
    *,
    model_name: str,
    language: str | None = None,
    audio_id: str | None = None,
) -> dict[str, Any]:
    """
    Transcribe one WAV file and return Bronze-compatible JSON.

    The returned structure is intentionally simple and compatible with
    scripts/generate_reconciliation_gradio_input.py.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _load_nemo_asr_model(model_name)

    try:
        results = model.transcribe(
            [str(audio_path)],
            timestamps=True,
            return_hypotheses=True,
        )
    except TypeError:
        results = model.transcribe([str(audio_path)])

    hypothesis = results[0] if isinstance(results, list) else results
    words = _extract_words_from_hypothesis(hypothesis)

    return {
        "schema_version": "bronze_transcript_v1",
        "audio_id": audio_id or audio_path.stem,
        "audio_path": str(audio_path),
        "model_name": model_name,
        "language": language,
        "words": words,
    }
