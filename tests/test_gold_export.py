import json

from src.annotation.gold_export import (
    SCHEMA_VERSION,
    export_gold_annotations,
    summarize_gold_items,
    to_gold_item,
)


def make_review_item(row_id="row_1"):
    return {
        "row_id": row_id,
        "audio_id": "audio001",
        "word_index": 0,
        "global_start": 0.0,
        "global_end": 0.5,
        "local_start": 0.0,
        "local_end": 0.5,
        "bronze_ar_text": "مرحبا",
        "bronze_en_text": "",
        "selected_language": "en-US",
        "corrected_text": "hello",
        "original_selected_language": "ar-AR",
        "original_corrected_text": "مرحبا",
        "language_hypotheses": [],
        "reconciliation_flags": ["language_conflict"],
        "left_context": ["before"],
        "right_context": ["after"],
        "review_status": "reviewed",
        "reviewed_at": "2026-07-07T00:00:00Z",
        "reviewer_id": "tester",
        "training_span": {
            "span_id": None,
            "language": "en-US",
            "start": 0.0,
            "end": 0.5,
            "text": "hello",
        },
        "source": {
            "schema_version": "gradio_reconciliation_input_v1",
            "source_index": 0,
        },
    }


def test_summarize_gold_items_counts_changes():
    summary = summarize_gold_items([make_review_item()])

    assert summary["total_rows"] == 1
    assert summary["reviewed_rows"] == 1
    assert summary["flagged_rows"] == 1
    assert summary["language_changed_rows"] == 1
    assert summary["text_changed_rows"] == 1


def test_to_gold_item_preserves_bronze_and_training_span():
    gold_item = to_gold_item(make_review_item())

    assert gold_item["row_id"] == "row_1"
    assert gold_item["selected_language"] == "en-US"
    assert gold_item["corrected_text"] == "hello"
    assert gold_item["bronze"]["bronze_ar_text"] == "مرحبا"
    assert gold_item["bronze"]["original_selected_language"] == "ar-AR"
    assert gold_item["training_span"]["language"] == "en-US"
    assert gold_item["training_span"]["text"] == "hello"


def test_export_gold_annotations_writes_json_and_jsonl(tmp_path):
    json_path = tmp_path / "gold.json"
    jsonl_path = tmp_path / "gold.jsonl"

    document = export_gold_annotations(
        [make_review_item()],
        output_json_path=json_path,
        output_jsonl_path=jsonl_path,
        source_file="input.json",
    )

    assert document["schema_version"] == SCHEMA_VERSION
    assert document["source_file"] == "input.json"
    assert document["review_summary"]["reviewed_rows"] == 1

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["items"][0]["corrected_text"] == "hello"

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["bronze"]["bronze_ar_text"] == "مرحبا"
