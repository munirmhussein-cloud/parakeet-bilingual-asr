"""Export reviewed annotations into the gold annotation JSON format."""

import json
from pathlib import Path

from .gradio_data_model import AnnotationRecord

SCHEMA_VERSION = "gold_annotations_v1"


def export_gold_annotations(
    records: list[AnnotationRecord],
    output_path: str | Path = "annotations/gold/gold_annotations_v1.json",
) -> dict[str, object]:
    """Write reviewed records to a gold annotations JSON document."""

    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "record_count": len(records),
        "annotations": [record.to_dict() for record in records],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return document
