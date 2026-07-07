from src.annotation.gradio_data_model import AnnotationRecord
from src.annotation.review_store import ReviewStore


def test_review_store_round_trips_records(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    record = AnnotationRecord(
        segment_id="seg-1",
        audio_path="data/audio/example.wav",
        asr_text="raw",
        corrected_text="corrected",
        reviewer="tester",
    )

    store.append(record)

    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].segment_id == "seg-1"
    assert loaded[0].corrected_text == "corrected"
