"""Transcript export helpers."""

import json
from pathlib import Path
from typing import Any


def write_transcript_json(record: dict[str, Any], output_path: str | Path) -> None:
    """Write one transcript record as formatted JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
