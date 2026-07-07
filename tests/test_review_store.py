from src.annotation.review_store import ReviewStore


def make_review_item(row_id="row_1", flagged=False):
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
        "original_selected_language": "ar-AR",
        "original_corrected_text": "مرحبا",
        "language_hypotheses": [],
        "reconciliation_flags": ["language_conflict"] if flagged else [],
        "left_context": [],
        "right_context": [],
        "review_status": "unreviewed",
        "reviewed_at": None,
        "reviewer_id": None,
        "training_span": {
            "span_id": None,
            "language": "ar-AR",
            "start": 0.0,
            "end": 0.5,
            "text": "مرحبا",
        },
        "source": {
            "schema_version": "gradio_reconciliation_input_v1",
            "source_index": 0,
        },
    }


def test_save_review_updates_item_snapshot_and_event_log(tmp_path):
    progress_path = tmp_path / "progress.json"
    events_path = tmp_path / "events.jsonl"
    store = ReviewStore(progress_path, events_path, reviewer_id="tester")

    items = [make_review_item(flagged=True)]

    updated = store.save_review(
        items,
        0,
        selected_language="en-US",
        corrected_text="hello",
    )

    assert updated["selected_language"] == "en-US"
    assert updated["corrected_text"] == "hello"
    assert updated["review_status"] == "reviewed"
    assert updated["reviewer_id"] == "tester"
    assert updated["training_span"]["language"] == "en-US"
    assert updated["training_span"]["text"] == "hello"

    assert progress_path.exists()
    assert events_path.exists()
    assert len(events_path.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_overlay_progress_restores_saved_review(tmp_path):
    progress_path = tmp_path / "progress.json"
    events_path = tmp_path / "events.jsonl"
    store = ReviewStore(progress_path, events_path, reviewer_id="tester")

    items = [make_review_item("row_1")]
    store.save_review(
        items,
        0,
        selected_language="en-US",
        corrected_text="hello",
    )

    fresh_items = [make_review_item("row_1")]
    restored = store.overlay_progress(fresh_items)

    assert restored[0]["selected_language"] == "en-US"
    assert restored[0]["corrected_text"] == "hello"
    assert restored[0]["review_status"] == "reviewed"


def test_get_resume_index_prioritizes_flagged_unreviewed():
    items = [
        {**make_review_item("row_1"), "review_status": "reviewed"},
        make_review_item("row_2", flagged=True),
        make_review_item("row_3"),
    ]

    assert ReviewStore.get_resume_index(items) == 1


def test_get_resume_index_falls_back_to_first_unreviewed():
    items = [
        {**make_review_item("row_1"), "review_status": "reviewed"},
        make_review_item("row_2"),
    ]

    assert ReviewStore.get_resume_index(items) == 1


def test_get_resume_index_returns_last_when_all_reviewed():
    items = [
        {**make_review_item("row_1"), "review_status": "reviewed"},
        {**make_review_item("row_2"), "review_status": "reviewed"},
    ]

    assert ReviewStore.get_resume_index(items) == 1
