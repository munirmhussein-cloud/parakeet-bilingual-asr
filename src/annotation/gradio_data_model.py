"""Data model objects used by the minimal Gradio annotation app."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class TranscriptSegment:
    """A single ASR transcript segment queued for human review."""

    segment_id: str
    audio_path: str
    asr_text: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    language: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        """Build a transcript segment from a JSON-compatible dictionary."""

        return cls(
            segment_id=str(data["segment_id"]),
            audio_path=str(data["audio_path"]),
            asr_text=str(data.get("asr_text", "")),
            start_seconds=data.get("start_seconds"),
            end_seconds=data.get("end_seconds"),
            language=data.get("language"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


@dataclass(frozen=True)
class AnnotationRecord:
    """Human review output for a transcript segment."""

    segment_id: str
    audio_path: str
    asr_text: str
    corrected_text: str
    reviewer: str
    decision: str = "accepted"
    language: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_segment(
        cls,
        segment: TranscriptSegment,
        *,
        corrected_text: str,
        reviewer: str,
        decision: str = "accepted",
        notes: str = "",
    ) -> "AnnotationRecord":
        """Create an annotation record from a transcript segment."""

        return cls(
            segment_id=segment.segment_id,
            audio_path=segment.audio_path,
            asr_text=segment.asr_text,
            corrected_text=corrected_text,
            reviewer=reviewer,
            decision=decision,
            language=segment.language,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            notes=notes,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotationRecord":
        """Build an annotation record from a JSON-compatible dictionary."""

        return cls(
            segment_id=str(data["segment_id"]),
            audio_path=str(data["audio_path"]),
            asr_text=str(data.get("asr_text", "")),
            corrected_text=str(data.get("corrected_text", "")),
            reviewer=str(data.get("reviewer", "")),
            decision=str(data.get("decision", "accepted")),
            language=data.get("language"),
            start_seconds=data.get("start_seconds"),
            end_seconds=data.get("end_seconds"),
            notes=str(data.get("notes", "")),
            created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)
