"""Audio metadata helpers for bilingual ASR datasets."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioMetadata:
    """Basic metadata for one audio file."""

    path: Path
    duration_seconds: float | None = None
    sample_rate_hz: int | None = None
    language: str | None = None


def build_metadata(path: str | Path, *, language: str | None = None) -> AudioMetadata:
    """Create an audio metadata record for a local path."""

    return AudioMetadata(path=Path(path), language=language)
