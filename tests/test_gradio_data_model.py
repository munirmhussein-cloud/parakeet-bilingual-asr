from src.annotation.gradio_data_model import AnnotationRecord, TranscriptSegment


def test_annotation_record_from_segment_preserves_segment_fields():
    segment = TranscriptSegment(
        segment_id="seg-1",
        audio_path="data/audio/example.wav",
        asr_text="hello world",
        start_seconds=1.0,
        end_seconds=2.5,
        language="en",
    )

    record = AnnotationRecord.from_segment(
        segment,
        corrected_text="hello, world",
        reviewer="tester",
        decision="corrected",
    )

    assert record.segment_id == "seg-1"
    assert record.audio_path == "data/audio/example.wav"
    assert record.asr_text == "hello world"
    assert record.corrected_text == "hello, world"
    assert record.reviewer == "tester"
    assert record.decision == "corrected"
    assert record.language == "en"
