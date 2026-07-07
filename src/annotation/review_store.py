"""
Sprint 2 review persistence.

Provides restart-safe load/resume/save behavior for normalized word-level
ReviewItem dictionaries.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .gradio_data_model import VALID_LANGUAGES, utc_now_iso


DEFAULT_PROGRESS_PATH = "annotations/progress/gradio_review_progress_v1.json"
DEFAULT_EVENTS_PATH = "annotations/progress/gradio_review_events_v1.jsonl"


class ReviewStore:
    """Persist and resume word-level Gradio annotation progress."""

    def __init__(
        self,
        progress_path: str | Path = DEFAULT_PROGRESS_PATH,
        events_path: str | Path = DEFAULT_EVENTS_PATH,
        reviewer_id: str | None = None,
    ) -> None:
        self.progress_path = Path(progress_path)
        self.events_path = Path(events_path)
        self.reviewer_id = reviewer_id

        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def load_progress(self) -> dict[str, Any] | None:
        """Load saved progress snapshot if it exists."""
        if not self.progress_path.exists():
            return None

        with self.progress_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def overlay_progress(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Overlay saved reviewer edits onto normalized source items by row_id."""
        progress = self.load_progress()
        if not progress:
            return deepcopy(items)

        saved_items = progress.get("items", [])
        saved_by_row_id = {
            item["row_id"]: item
            for item in saved_items
            if isinstance(item, dict) and "row_id" in item
        }

        merged: list[dict[str, Any]] = []

        for item in items:
            row_id = item["row_id"]
            current = deepcopy(item)

            if row_id in saved_by_row_id:
                saved = saved_by_row_id[row_id]
                current.update(saved)

                if "training_span" in current and isinstance(current["training_span"], dict):
                    current["training_span"]["language"] = current.get("selected_language")
                    current["training_span"]["text"] = current.get("corrected_text")

            merged.append(current)

        return merged

    def save_snapshot(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Write full progress snapshot atomically enough for Colab use."""
        reviewed = [item for item in items if item.get("review_status") == "reviewed"]

        snapshot = {
            "schema_version": "gradio_review_progress_v1",
            "updated_at": utc_now_iso(),
            "reviewer_id": self.reviewer_id,
            "summary": {
                "total_items": len(items),
                "reviewed_items": len(reviewed),
                "unreviewed_items": len(items) - len(reviewed),
                "flagged_items": sum(1 for item in items if item.get("reconciliation_flags")),
                "reviewed_flagged_items": sum(
                    1
                    for item in items
                    if item.get("reconciliation_flags")
                    and item.get("review_status") == "reviewed"
                ),
            },
            "items": items,
        }

        tmp_path = self.progress_path.with_suffix(self.progress_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self.progress_path)

        return snapshot

    def append_event(
        self,
        *,
        row_id: str,
        selected_language: str,
        corrected_text: str,
        event_type: str = "review_saved",
    ) -> dict[str, Any]:
        """Append one review event to JSONL audit log."""
        event = {
            "schema_version": "gradio_review_event_v1",
            "event_type": event_type,
            "created_at": utc_now_iso(),
            "reviewer_id": self.reviewer_id,
            "row_id": row_id,
            "selected_language": selected_language,
            "corrected_text": corrected_text,
        }

        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

        return event

    def save_review(
        self,
        items: list[dict[str, Any]],
        index: int,
        *,
        selected_language: str,
        corrected_text: str,
    ) -> dict[str, Any]:
        """Apply one reviewer edit, append event, and save snapshot."""
        if selected_language not in VALID_LANGUAGES:
            raise ValueError(f"Invalid selected_language: {selected_language}")

        if not isinstance(corrected_text, str):
            raise TypeError("corrected_text must be a string")

        if index < 0 or index >= len(items):
            raise IndexError(f"Review index out of range: {index}")

        item = items[index]
        item["selected_language"] = selected_language
        item["corrected_text"] = corrected_text
        item["review_status"] = "reviewed"
        item["reviewed_at"] = utc_now_iso()
        item["reviewer_id"] = self.reviewer_id

        if "training_span" in item and isinstance(item["training_span"], dict):
            item["training_span"]["language"] = selected_language
            item["training_span"]["text"] = corrected_text

        self.append_event(
            row_id=item["row_id"],
            selected_language=selected_language,
            corrected_text=corrected_text,
        )

        self.save_snapshot(items)

        return item

    @staticmethod
    def get_resume_index(items: list[dict[str, Any]]) -> int:
        """
        Resume at first flagged unreviewed item.
        If none remain, resume at first unreviewed item.
        If all reviewed, return final item index.
        """
        if not items:
            return 0

        for index, item in enumerate(items):
            if item.get("reconciliation_flags") and item.get("review_status") != "reviewed":
                return index

        for index, item in enumerate(items):
            if item.get("review_status") != "reviewed":
                return index

        return len(items) - 1

    @staticmethod
    def get_next_index(items: list[dict[str, Any]], current_index: int) -> int:
        """Return next row index without exceeding bounds."""
        if not items:
            return 0
        return min(current_index + 1, len(items) - 1)
