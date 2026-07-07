"""Minimal Gradio annotator for bilingual ASR transcript review."""

from pathlib import Path

from src.annotation import AnnotationRecord, ReviewStore, TranscriptSegment

DEFAULT_SEGMENT = TranscriptSegment(
    segment_id="sample-001",
    audio_path="data/audio/sample.wav",
    asr_text="Sample ASR transcript text.",
    language="en",
)


def save_review(
    corrected_text: str,
    reviewer: str,
    decision: str = "accepted",
    notes: str = "",
    store_path: str = "annotations/progress/reviews.jsonl",
) -> str:
    """Save a review for the currently loaded sample segment."""

    record = AnnotationRecord.from_segment(
        DEFAULT_SEGMENT,
        corrected_text=corrected_text,
        reviewer=reviewer,
        decision=decision,
        notes=notes,
    )
    ReviewStore(store_path).append(record)
    return f"Saved review for {record.segment_id} to {Path(store_path)}"


def build_app():
    """Build the Gradio Blocks app lazily so tests do not require Gradio."""

    import gradio as gr

    with gr.Blocks(title="Bilingual ASR Annotator") as demo:
        gr.Markdown("# Bilingual ASR Annotator")
        gr.Markdown("Review ASR text and save corrected gold annotations.")
        asr_text = gr.Textbox(label="ASR transcript", value=DEFAULT_SEGMENT.asr_text, interactive=False)
        corrected_text = gr.Textbox(label="Corrected transcript", value=DEFAULT_SEGMENT.asr_text)
        reviewer = gr.Textbox(label="Reviewer", value="reviewer")
        decision = gr.Radio(["accepted", "corrected", "rejected"], value="accepted", label="Decision")
        notes = gr.Textbox(label="Notes", value="")
        output = gr.Textbox(label="Status")
        save_button = gr.Button("Save review")
        save_button.click(save_review, [corrected_text, reviewer, decision, notes], output)
        asr_text.change(lambda value: value, asr_text, corrected_text)
    return demo


if __name__ == "__main__":
    build_app().launch()
