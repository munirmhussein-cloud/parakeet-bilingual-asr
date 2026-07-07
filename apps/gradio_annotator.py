"""
Minimal Sprint 2 Gradio annotator.

Loads gradio_reconciliation_input_v1.json, overlays saved progress,
supports language toggle + corrected_text edits, and exports Gold JSON/JSONL.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gradio as gr

from src.annotation.gold_export import export_gold_annotations
from src.annotation.gradio_data_model import normalize_reconciliation_input
from src.annotation.review_store import ReviewStore


def format_context(item: dict[str, Any]) -> str:
    left = " ".join(str(x) for x in item.get("left_context", []))
    right = " ".join(str(x) for x in item.get("right_context", []))
    current = item.get("corrected_text", "")
    return f"{left}  >>>  {current}  <<<  {right}".strip()


def format_flags(item: dict[str, Any]) -> str:
    flags = item.get("reconciliation_flags", [])
    if flags:
        return "⚠️ FLAGGED: " + ", ".join(str(flag) for flag in flags)
    return "No flags"


def format_timestamp(item: dict[str, Any]) -> str:
    return (
        f"global: {item.get('global_start')} → {item.get('global_end')} | "
        f"local: {item.get('local_start')} → {item.get('local_end')}"
    )


def render_item(items: list[dict[str, Any]], index: int):
    if not items:
        return (
            0,
            "0 / 0",
            "No items loaded",
            "",
            "",
            "",
            "",
            "",
            "ar-AR",
            "",
            "",
        )

    index = max(0, min(index, len(items) - 1))
    item = items[index]

    progress = f"{index + 1} / {len(items)} | status: {item.get('review_status', 'unreviewed')}"

    return (
        index,
        progress,
        format_flags(item),
        item.get("row_id", ""),
        format_timestamp(item),
        format_context(item),
        item.get("bronze_ar_text", ""),
        item.get("bronze_en_text", ""),
        item.get("selected_language", "ar-AR"),
        item.get("corrected_text", ""),
        item.get("reviewer_id") or "",
    )


def build_app(
    input_path: str | Path,
    progress_path: str | Path,
    events_path: str | Path,
    gold_json_path: str | Path,
    gold_jsonl_path: str | Path,
    reviewer_id: str | None = None,
):
    normalized = normalize_reconciliation_input(input_path)
    store = ReviewStore(
        progress_path=progress_path,
        events_path=events_path,
        reviewer_id=reviewer_id,
    )

    items = store.overlay_progress(normalized["items"])
    start_index = store.get_resume_index(items)

    def load_initial():
        return render_item(items, start_index)

    def previous(index):
        return render_item(items, max(index - 1, 0))

    def skip(index):
        return render_item(items, store.get_next_index(items, index))

    def save(index, selected_language, corrected_text):
        store.save_review(
            items,
            index,
            selected_language=selected_language,
            corrected_text=corrected_text,
        )
        return render_item(items, index)

    def save_and_next(index, selected_language, corrected_text):
        store.save_review(
            items,
            index,
            selected_language=selected_language,
            corrected_text=corrected_text,
        )
        next_index = store.get_next_index(items, index)
        return render_item(items, next_index)

    def export_gold():
        document = export_gold_annotations(
            items,
            output_json_path=gold_json_path,
            output_jsonl_path=gold_jsonl_path,
            source_file=str(input_path),
        )
        summary = document["review_summary"]
        return (
            f"Exported Gold files:\n"
            f"{gold_json_path}\n"
            f"{gold_jsonl_path}\n\n"
            f"Reviewed rows: {summary['reviewed_rows']} / {summary['total_rows']}\n"
            f"Flagged rows: {summary['flagged_rows']}\n"
            f"Language changes: {summary['language_changed_rows']}\n"
            f"Text changes: {summary['text_changed_rows']}"
        )

    with gr.Blocks(title="Sprint 2 Minimal Gradio Annotator") as demo:
        gr.Markdown("# Sprint 2 Minimal Gradio Annotator")
        gr.Markdown(
            "Review one word-level reconciliation row at a time. "
            "Only `selected_language` and `corrected_text` are editable."
        )

        index_state = gr.State(start_index)

        progress = gr.Textbox(label="Progress", interactive=False)
        flag_status = gr.Textbox(label="Flag Status", interactive=False)
        row_id = gr.Textbox(label="Row ID", interactive=False)
        timestamp = gr.Textbox(label="Timestamps", interactive=False)
        context = gr.Textbox(label="Context", interactive=False)

        with gr.Row():
            bronze_ar = gr.Textbox(label="Bronze AR", interactive=False)
            bronze_en = gr.Textbox(label="Bronze EN", interactive=False)

        selected_language = gr.Radio(
            choices=["ar-AR", "en-US"],
            label="Selected Language",
        )
        corrected_text = gr.Textbox(label="Corrected Text")
        reviewer = gr.Textbox(label="Reviewer", interactive=False)

        with gr.Row():
            previous_btn = gr.Button("Previous")
            skip_btn = gr.Button("Skip")
            save_btn = gr.Button("Save")
            save_next_btn = gr.Button("Save & Next", variant="primary")
            export_btn = gr.Button("Export Gold")

        export_status = gr.Textbox(label="Export Status", interactive=False)

        item_outputs = [
            index_state,
            progress,
            flag_status,
            row_id,
            timestamp,
            context,
            bronze_ar,
            bronze_en,
            selected_language,
            corrected_text,
            reviewer,
        ]

        demo.load(fn=load_initial, outputs=item_outputs)

        previous_btn.click(fn=previous, inputs=index_state, outputs=item_outputs)
        skip_btn.click(fn=skip, inputs=index_state, outputs=item_outputs)
        save_btn.click(
            fn=save,
            inputs=[index_state, selected_language, corrected_text],
            outputs=item_outputs,
        )
        save_next_btn.click(
            fn=save_and_next,
            inputs=[index_state, selected_language, corrected_text],
            outputs=item_outputs,
        )
        export_btn.click(fn=export_gold, outputs=export_status)

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sprint 2 Gradio annotator.")
    parser.add_argument(
        "--input",
        default="data/annotations/gradio_reconciliation_input_v1.json",
        help="Path to gradio_reconciliation_input_v1.json",
    )
    parser.add_argument(
        "--progress",
        default="annotations/progress/gradio_review_progress_v1.json",
        help="Path to progress snapshot JSON",
    )
    parser.add_argument(
        "--events",
        default="annotations/progress/gradio_review_events_v1.jsonl",
        help="Path to review event JSONL",
    )
    parser.add_argument(
        "--gold-json",
        default="annotations/gold/gold_annotations_v1.json",
        help="Gold JSON output path",
    )
    parser.add_argument(
        "--gold-jsonl",
        default="annotations/gold/gold_annotations_v1.jsonl",
        help="Gold JSONL output path",
    )
    parser.add_argument("--reviewer-id", default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    demo = build_app(
        input_path=args.input,
        progress_path=args.progress,
        events_path=args.events,
        gold_json_path=args.gold_json,
        gold_jsonl_path=args.gold_jsonl,
        reviewer_id=args.reviewer_id,
    )
    demo.launch(share=args.share, debug=args.debug)


if __name__ == "__main__":
    main()
