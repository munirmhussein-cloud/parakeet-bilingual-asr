import json
from pathlib import Path

import pytest

from src.annotation.gradio_data_model import (
    normalize_reconciliation_input,
    normalize_review_item,
    validate_reconciliation_input,
)


def make_item(row_id="row_1"):
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
        "selected_language": "ar-AR",
        "corrected_text": "مرحبا",
        "reconciliation_flags": ["language_conflict"],
        "left_context": ["before"],
        "right_context": ["after"],
    }


def write_input(path: Path, items):
    path.write_text(
        json.dumps(
            {
                "schema_version": "gradio_reconciliation_input_v1",
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_validate_reconciliation_input_passes(tmp_path):
    input_path = tmp_path / "input.json"
    write_input(input_path, [make_item()])

    _, summary = validate_reconciliation_input(input_path)

    assert summary["errors"] == []
    assert summary["total_items"] == 1
    assert summary["flagged_items"] == 1
    assert summary["language_counts"]["ar-AR"] == 1


def test_normalize_review_item_preserves_original_values():
    item = make_item()

    normalized = normalize_review_item(item, 0)

    assert normalized["row_id"] == "row_1"
    assert normalized["selected_language"] == "ar-AR"
    assert normalized["corrected_text"] == "مرحبا"
    assert normalized["original_selected_language"] == "ar-AR"
    assert normalized["original_corrected_text"] == "مرحبا"
    assert normalized["review_status"] == "unreviewed"
    assert normalized["training_span"]["language"] == "ar-AR"
    assert normalized["training_span"]["text"] == "مرحبا"


def test_normalize_reconciliation_input_summary(tmp_path):
    input_path = tmp_path / "input.json"
    write_input(
        input_path,
        [
            make_item("row_1"),
            {
                **make_item("row_2"),
                "selected_language": "en-US",
                "corrected_text": "hello",
            },
        ],
    )

    normalized = normalize_reconciliation_input(input_path)

    assert normalized["schema_version"] == "normalized_review_items_v1"
    assert normalized["summary"]["total_items"] == 2
    assert normalized["summary"]["flagged_items"] == 2
    assert normalized["summary"]["language_counts"]["ar-AR"] == 1
    assert normalized["summary"]["language_counts"]["en-US"] == 1


def test_duplicate_row_id_fails(tmp_path):
    input_path = tmp_path / "input.json"
    write_input(input_path, [make_item("dup"), make_item("dup")])

    with pytest.raises(ValueError, match="Duplicate row_id"):
        normalize_reconciliation_input(input_path)


def test_invalid_language_fails():
    item = make_item()
    item["selected_language"] = "fr-FR"

    with pytest.raises(ValueError, match="invalid selected_language"):
        normalize_review_item(item, 0)


def test_missing_required_field_fails():
    item = make_item()
    del item["corrected_text"]

    with pytest.raises(ValueError, match="missing required fields"):
        normalize_review_item(item, 0)
