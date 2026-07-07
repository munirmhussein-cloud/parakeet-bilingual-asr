"""Append-only storage for in-progress annotation reviews."""

import json
from pathlib import Path
from typing import Iterable

from .gradio_data_model import AnnotationRecord


class ReviewStore:
    """Persist annotation records as JSON Lines."""

    def __init__(self, path: str | Path = "annotations/progress/reviews.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: AnnotationRecord) -> None:
        """Append one annotation record to the store."""

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def load(self) -> list[AnnotationRecord]:
        """Load all annotation records from the store."""

        if not self.path.exists():
            return []

        records: list[AnnotationRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(AnnotationRecord.from_dict(json.loads(line)))
        return records

    def extend(self, records: Iterable[AnnotationRecord]) -> None:
        """Append multiple records to the store."""

        for record in records:
            self.append(record)
