"""
Sprint 3 Gradio annotator.

Primary Annotation Mode:
- table-style bulk review
- Bronze EN
- Bronze AR
- corrected text
- language label
- bulk save

Secondary Review Mode:
- original Sprint 2 one-row-at-a-time workflow
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from src.annotation.gold_export import export_gold_annotations
from src.annotation.gradio_data_model import normalize_reconciliation_input
from src.annotation.review_store import ReviewStore


LANGUAGE_CHOICES = ["ar-AR", "en-US"]


def detect_language_from_script(text: str) -> str:
    text = str(text or "")
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))

    if arabic_chars > latin_chars:
        return "ar-AR"
    if latin_chars > arabic_chars:
        return "en-US"
    return "ar-AR"



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


def ms_to_mmss(value: Any) -> str:
    try:
        total_seconds = float(value) / 1000.0
    except (TypeError, ValueError):
        return "?:??"

    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)

    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"


def format_timestamp(item: dict[str, Any]) -> str:
    local_start = item.get("local_start")
    local_end = item.get("local_end")

    return (
        f"segment: {ms_to_mmss(local_start)} → {ms_to_mmss(local_end)} | "
        f"global: {item.get('global_start')} → {item.get('global_end')} | "
        f"local: {local_start} → {local_end}"
    )


def items_to_dataframe(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, item in enumerate(items):
        rows.append(
            {
                "index": i,
                "row_id": item.get("row_id", ""),
                "Bronze EN": item.get("bronze_en_text", ""),
                "Bronze AR": item.get("bronze_ar_text", ""),
                "Corrected Text": item.get("corrected_text", ""),
                "Language": item.get("selected_language", "ar-AR"),
                "Status": item.get("review_status", "unreviewed"),
                "Flags": ", ".join(str(x) for x in item.get("reconciliation_flags", [])),
            }
        )
    return pd.DataFrame(rows)


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



def get_source_audio(normalized: dict[str, Any], items: list[dict[str, Any]]) -> str | None:
    source_audio = normalized.get("source_audio")
    if source_audio:
        return str(source_audio)
    if items:
        audio_path = items[0].get("audio_path")
        if audio_path:
            return str(audio_path)
    return None


def get_row_choices(items: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("row_id", f"row_{i:05d}")) for i, item in enumerate(items)]


def row_id_to_index(items: list[dict[str, Any]], row_id: str) -> int:
    for i, item in enumerate(items):
        if str(item.get("row_id", "")) == str(row_id):
            return i
    return 0


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
    source_audio = get_source_audio(normalized, items)
    row_choices = get_row_choices(items)

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

    def load_table():
        return items_to_dataframe(items), f"Loaded {len(items)} rows."

    def save_table(df):
        if df is None:
            return items_to_dataframe(items), "No table data received."

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        saved = 0
        errors = []

        previous_language = "ar-AR"

        for _, row in df.iterrows():
            try:
                index = int(row["index"])
                corrected = str(row.get("Corrected Text", "") or "").strip()
                language = str(row.get("Language", "") or "").strip()

                if corrected:
                    if language not in LANGUAGE_CHOICES:
                        language = detect_language_from_script(corrected)
                    previous_language = language
                else:
                    # Blank corrected text means this row is merged into the
                    # previous non-blank row and should inherit its language.
                    language = previous_language

                if language not in LANGUAGE_CHOICES:
                    errors.append(f"Row {index}: invalid language `{language}`")
                    continue

                original = items[index]
                changed = (
                    corrected != str(original.get("corrected_text", "") or "").strip()
                    or language != str(original.get("selected_language", "") or "")
                    or original.get("review_status") != "reviewed"
                )

                if changed:
                    store.save_review(
                        items,
                        index,
                        selected_language=language,
                        corrected_text=corrected,
                    )
                    items[index]["review_status"] = "reviewed"
                    saved += 1

            except Exception as exc:
                errors.append(str(exc))

        msg = f"Bulk save complete. Updated rows: {saved}."
        if errors:
            msg += "\nErrors:\n" + "\n".join(errors[:10])

        return items_to_dataframe(items), msg

    def auto_detect_languages(df):
        if df is None:
            return items_to_dataframe(items), "No table data received."

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        df = df.copy()
        updated = 0
        previous_language = "ar-AR"

        for i, row in df.iterrows():
            corrected = str(row.get("Corrected Text", "") or "").strip()

            if corrected:
                detected = detect_language_from_script(corrected)
                previous_language = detected
            else:
                # Blank corrected text means this row is merged into the previous
                # non-blank row, so it inherits the previous language.
                detected = previous_language

            if str(row.get("Language", "") or "") != detected:
                df.at[i, "Language"] = detected
                updated += 1

        return df, f"Auto-detected languages for {len(df)} rows. Blank rows inherit the previous language. Updated language cells: {updated}."

    def load_initial():
        return render_item(items, start_index)

    def previous(index):
        return render_item(items, max(index - 1, 0))

    def skip(index):
        return render_item(items, store.get_next_index(items, index))

    def next_row(index):
        return render_item(items, min(index + 1, len(items) - 1))

    def jump_to_row(row_id):
        return render_item(items, row_id_to_index(items, row_id))

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

    with gr.Blocks(title="Bilingual Annotation Platform") as demo:
        gr.Markdown("# Bilingual Annotation Platform")
        gr.Markdown(
            "Sprint 3 interface: bulk Annotation Mode first, detailed Review Mode second."
        )

        with gr.Tab("Annotation Mode"):
            gr.Markdown(
                "Edit rows in bulk. Preserve Bronze EN/AR, update Corrected Text and Language, then save all changes."
            )

            annotation_table = gr.Dataframe(
                headers=[
                    "index",
                    "row_id",
                    "Bronze EN",
                    "Bronze AR",
                    "Corrected Text",
                    "Language",
                    "Status",
                    "Flags",
                ],
                datatype=["number", "str", "str", "str", "str", "str", "str", "str"],
                interactive=True,
                wrap=True,
                label="Segment Annotation Table",
            )

            with gr.Row():
                reload_table_btn = gr.Button("Reload Table")
                auto_detect_btn = gr.Button("Auto-detect Languages")
                save_table_btn = gr.Button("Bulk Save", variant="primary")
                export_table_btn = gr.Button("Export Gold")

            table_status = gr.Textbox(label="Annotation Mode Status", interactive=False)

            demo.load(fn=load_table, outputs=[annotation_table, table_status])
            reload_table_btn.click(fn=load_table, outputs=[annotation_table, table_status])
            auto_detect_btn.click(
                fn=auto_detect_languages,
                inputs=annotation_table,
                outputs=[annotation_table, table_status],
            )
            save_table_btn.click(
                fn=save_table,
                inputs=annotation_table,
                outputs=[annotation_table, table_status],
            )
            export_table_btn.click(fn=export_gold, outputs=table_status)

        with gr.Tab("Review Mode"):
            gr.Markdown(
                "Detailed one-row review mode retained from Sprint 2 for edge cases."
            )

            index_state = gr.State(start_index)

            gr.Markdown("## Row Navigation")
            row_dropdown = gr.Dropdown(
                choices=row_choices,
                value=row_choices[start_index] if row_choices else None,
                label="Dropdown: row_id",
            )

            with gr.Row():
                previous_btn = gr.Button("Previous Row")
                jump_btn = gr.Button("Jump to Row", variant="primary")
                next_row_btn = gr.Button("Next Row")

            gr.Markdown("---")

            row_id = gr.Textbox(label="Row ID", interactive=False)
            timestamp = gr.Textbox(label="Timestamps", interactive=False)
            context = gr.Textbox(label="Context", interactive=False)

            gr.Markdown("---")

            audio_player = gr.Audio(
                value=source_audio,
                label="Full Segment Audio Player",
                interactive=False,
            )

            gr.Markdown("---")

            corrected_text = gr.Textbox(label="Corrected Text")

            gr.Markdown("---")

            with gr.Row():
                bronze_ar = gr.Textbox(label="Bronze AR", interactive=False)
                bronze_en = gr.Textbox(label="Bronze EN", interactive=False)

            gr.Markdown("---")

            selected_language = gr.Radio(
                choices=LANGUAGE_CHOICES,
                label="Selected Language",
            )

            gr.Markdown("---")

            with gr.Row():
                previous_segment_btn = gr.Button("Previous Segment", interactive=False)
                next_segment_btn = gr.Button("Next Segment", interactive=False)

            audio_source = gr.Textbox(
                value=source_audio or "",
                label="Audio source",
                interactive=False,
            )
            progress = gr.Textbox(label="Progress", interactive=False)
            reviewer = gr.Textbox(label="Reviewer", interactive=False)

            gr.Markdown("---")

            with gr.Row():
                save_btn = gr.Button("Save")
                save_next_btn = gr.Button("Save & Next", variant="primary")
                export_btn = gr.Button("Export Gold")

            export_status = gr.Textbox(label="Export Status", interactive=False)
            flag_status = gr.Textbox(visible=False)

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
            next_row_btn.click(fn=next_row, inputs=index_state, outputs=item_outputs)
            jump_btn.click(fn=jump_to_row, inputs=row_dropdown, outputs=item_outputs)

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
    parser = argparse.ArgumentParser(description="Run Sprint 3 Gradio annotator.")
    parser.add_argument(
        "--input",
        default="data/annotations/gradio_reconciliation_input_v1.json",
    )
    parser.add_argument(
        "--progress",
        default="annotations/progress/gradio_review_progress_v1.json",
    )
    parser.add_argument(
        "--events",
        default="annotations/progress/gradio_review_events_v1.jsonl",
    )
    parser.add_argument(
        "--gold-json",
        default="annotations/gold/gold_annotations_v1.json",
    )
    parser.add_argument(
        "--gold-jsonl",
        default="annotations/gold/gold_annotations_v1.jsonl",
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
