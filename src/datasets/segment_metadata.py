from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SegmentMetadataResolver:
    """
    Resolve authoritative segment metadata from a segment manifest JSONL.

    Segment manifest fields currently expected:
    - segment_id
    - audio_filepath
    - duration

    Additional fields are preserved for future use:
    - source_audio_id
    - start_time
    - end_time
    - speaker_id
    - split
    - checksum / sha256
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.by_segment_id: dict[str, dict[str, Any]] = {}
        self.by_audio_filepath: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing segment manifest: {self.manifest_path}")

        for line_no, line in enumerate(
            self.manifest_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue

            row = json.loads(line)

            segment_id = row.get("segment_id")
            audio_filepath = row.get("audio_filepath")

            if not segment_id:
                raise ValueError(f"Manifest line {line_no} missing segment_id")
            if not audio_filepath:
                raise ValueError(f"Manifest line {line_no} missing audio_filepath")

            self.by_segment_id[str(segment_id)] = row
            self.by_audio_filepath[str(audio_filepath)] = row

    def resolve(
        self,
        *,
        segment_id: str | None = None,
        audio_filepath: str | None = None,
    ) -> dict[str, Any] | None:
        if segment_id and segment_id in self.by_segment_id:
            return self.by_segment_id[segment_id]

        if audio_filepath and audio_filepath in self.by_audio_filepath:
            return self.by_audio_filepath[audio_filepath]

        return None

    def require(
        self,
        *,
        segment_id: str | None = None,
        audio_filepath: str | None = None,
    ) -> dict[str, Any]:
        row = self.resolve(segment_id=segment_id, audio_filepath=audio_filepath)
        if row is None:
            raise KeyError(
                f"No segment metadata found for "
                f"segment_id={segment_id!r}, audio_filepath={audio_filepath!r}"
            )
        return row
