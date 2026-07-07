import json

from scripts.generate_reconciliation_gradio_input import (
    extract_words,
    generate_gradio_input,
    generate_gradio_items,
)


def test_extract_words_from_top_level_words():
    bronze = {
        "audio_id": "audio001",
        "words": [
            {"text": "مرحبا", "start": 0.0, "end": 0.5},
        ],
    }

    words = extract_words(bronze)

    assert len(words) == 1
    assert words[0]["text"] == "مرحبا"


def test_extract_words_from_segments():
    bronze = {
        "segments": [
            {
                "segment_id": "seg001",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.5},
                ],
            }
        ]
    }

    words = extract_words(bronze)

    assert len(words) == 1
    assert words[0]["word"] == "hello"
    assert words[0]["segment_id"] == "seg001"


def test_generate_gradio_items_pairs_bronze_outputs():
    bronze_ar = {
        "audio_id": "audio001",
        "words": [
            {"text": "مرحبا", "start": 0.0, "end": 0.5},
            {"text": "كيف", "start": 0.6, "end": 1.0},
        ],
    }
    bronze_en = {
        "audio_id": "audio001",
        "words": [
            {"text": "hello", "start": 0.0, "end": 0.5},
            {"text": "", "start": 0.6, "end": 1.0},
        ],
    }

    items = generate_gradio_items(bronze_ar, bronze_en, audio_id="audio001")

    assert len(items) == 2
    assert items[0]["row_id"] == "audio001_word_00000"
    assert items[0]["bronze_ar_text"] == "مرحبا"
    assert items[0]["bronze_en_text"] == "hello"
    assert "dual_bronze_text" in items[0]["reconciliation_flags"]
    assert items[1]["selected_language"] == "ar-AR"


def test_generate_gradio_input_writes_file(tmp_path):
    bronze_ar_path = tmp_path / "bronze_ar.json"
    bronze_en_path = tmp_path / "bronze_en.json"
    output_path = tmp_path / "gradio.json"

    bronze_ar_path.write_text(
        json.dumps(
            {
                "audio_id": "audio001",
                "words": [{"text": "مرحبا", "start": 0.0, "end": 0.5}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    bronze_en_path.write_text(
        json.dumps(
            {
                "audio_id": "audio001",
                "words": [{"text": "hello", "start": 0.0, "end": 0.5}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    document = generate_gradio_input(
        bronze_ar_path,
        bronze_en_path,
        output_path,
        audio_id="audio001",
        audio_path="data/audio/audio001.wav",
    )

    assert output_path.exists()
    assert document["schema_version"] == "gradio_reconciliation_input_v1"
    assert document["items"][0]["audio_path"] == "data/audio/audio001.wav"
