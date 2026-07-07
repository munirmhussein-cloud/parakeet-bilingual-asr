import json

from src.annotation.gold_export import SCHEMA_VERSION, export_gold_annotations
from src.annotation.gradio_data_model import AnnotationRecord


def test_export_gold_annotations_writes_document(tmp_path):
    record = AnnotationRecord(
        segment_id="seg-1",
        audio_path="data/audio/example.wav",
        asr_text="raw",
        corrected_text="corrected",
        reviewer="tester",
    )
    output_path = tmp_path / "gold.json"

    document = export_gold_annotations([record], output_path)

    assert document["schema_version"] == SCHEMA_VERSION
    assert document["record_count"] == 1
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["annotations"][0]["corrected_text"] == "corrected"
