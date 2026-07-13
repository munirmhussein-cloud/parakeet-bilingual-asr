from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from src.annotation.gold_export import export_gold_annotations
from src.annotation.gradio_data_model import normalize_reconciliation_input
from src.annotation.review_store import ReviewStore


LANGUAGE_CHOICES = ["ar-AR", "en-US"]


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def segment_label(row: dict[str, Any], index: int) -> str:
    return row.get("segment_id") or Path(row["audio_filepath"]).stem or f"segment_{index:06d}"


def segment_number(label: str) -> str | None:
    match = re.search(r"seg_\d{6}", label)
    return match.group(0) if match else None


def find_reconciliation(reconciliation_dir: Path, label: str) -> Path:
    num = segment_number(label)
    candidates = []
    if num:
        candidates.extend(reconciliation_dir.glob(f"*{num}*_reconciliation.json"))
    candidates.extend(reconciliation_dir.glob(f"*{label}*_reconciliation.json"))

    if not candidates:
        raise FileNotFoundError(f"No reconciliation file found for segment: {label}")

    return sorted(candidates)[0]


def ms_to_seconds(value: Any) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    return float(value) / 1000.0


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}:{secs:05.2f}"


def highlighted_context(item: dict[str, Any]) -> str:
    left = " ".join(str(x) for x in item.get("left_context", []))
    current = str(item.get("corrected_text", ""))
    right = " ".join(str(x) for x in item.get("right_context", []))
    return (
        "<div style='font-size:18px;line-height:1.8'>"
        f"{left} "
        "<span style='"
        "display:inline-block;"
        "padding:3px 10px;"
        "border:2px solid currentColor;"
        "border-radius:999px;"
        "font-weight:700;"
        "background:rgba(128,128,128,0.18);"
        "box-shadow:0 0 0 2px rgba(128,128,128,0.12);"
        "'>"
        f"{current}"
        "</span>"
        f" {right}</div>"
    )


def timestamp_label(item: dict[str, Any]) -> str:
    local_start = item.get("local_start")
    local_end = item.get("local_end")
    seg_start_s = ms_to_seconds(local_start)
    seg_end_s = ms_to_seconds(local_end)
    return (
        f"segment: {format_time(seg_start_s)} → {format_time(seg_end_s)} | "
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


def detect_language_from_script(text: str) -> str:
    text = str(text or "")
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    if arabic_chars > latin_chars:
        return "ar-AR"
    if latin_chars > arabic_chars:
        return "en-US"
    return "ar-AR"


def make_word_clip(audio_path: str, item: dict[str, Any], clip_dir: Path) -> str | None:
    source = Path(audio_path)
    if not source.exists():
        return None

    start = ms_to_seconds(item.get("local_start"))
    end = ms_to_seconds(item.get("local_end"))
    duration = max(1.5, min(6.0, end - start + 1.0))

    clip_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", item.get("row_id", "word_clip"))
    out = clip_dir / f"{safe}.wav"

    if not out.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(max(0.0, start)),
                "-t",
                str(duration),
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(out),
            ],
            check=False,
        )

    return str(out) if out.exists() else None


def build_app(
    manifest_path: str | Path,
    reconciliation_dir: str | Path,
    progress_dir: str | Path = "data/annotations/progress_manifest_browser",
    events_dir: str | Path = "data/annotations/events_manifest_browser",
    gold_dir: str | Path = "data/annotations/gold_manifest_browser",
    clip_dir: str | Path = "data/annotations/playback_clips",
):
    manifest_rows = load_manifest(manifest_path)
    labels = [segment_label(row, i) for i, row in enumerate(manifest_rows)]
    reconciliation_dir = Path(reconciliation_dir)
    progress_dir = Path(progress_dir)
    events_dir = Path(events_dir)
    gold_dir = Path(gold_dir)
    clip_dir = Path(clip_dir)

    state = {
        "segment_index": 0,
        "row_index": 0,
        "items": [],
        "audio_path": None,
        "input_path": None,
        "store": None,
    }

    def load_segment(index: int):
        index = max(0, min(index, len(labels) - 1))
        label = labels[index]
        input_path = find_reconciliation(reconciliation_dir, label)
        normalized = normalize_reconciliation_input(input_path)

        audio_path = normalized.get("source_audio")
        if not audio_path and normalized["items"]:
            audio_path = normalized["items"][0].get("audio_path")
        if not audio_path:
            audio_path = manifest_rows[index].get("audio_filepath")

        progress_path = progress_dir / f"{label}.json"
        events_path = events_dir / f"{label}.jsonl"

        store = ReviewStore(progress_path=progress_path, events_path=events_path)
        items = store.overlay_progress(normalized["items"])

        state.update(
            {
                "segment_index": index,
                "row_index": store.get_resume_index(items),
                "items": items,
                "audio_path": audio_path,
                "input_path": str(input_path),
                "store": store,
            }
        )

        return render()

    def save_current(language: str, corrected: str):
        items = state["items"]
        if not items:
            return
        idx = state["row_index"]
        state["store"].save_review(
            items,
            idx,
            selected_language=language,
            corrected_text=corrected,
        )

    def render():
        items = state["items"]
        if not items:
            return (
                labels[0],
                [],
                None,
                "",
                "",
                "",
                "",
                "",
                "ar-AR",
                "",
                "",
                "",
                None,
                "",
            )

        idx = max(0, min(state["row_index"], len(items) - 1))
        state["row_index"] = idx
        item = items[idx]
        row_choices = [x["row_id"] for x in items]
        audio_path = state["audio_path"]
        word_clip = make_word_clip(audio_path, item, clip_dir) if audio_path else None

        progress = f"{idx + 1} / {len(items)} | status: {item.get('review_status', 'unreviewed')}"
        segment_progress = f"Segment {state['segment_index'] + 1} / {len(labels)}"

        return (
            labels[state["segment_index"]],
            gr.update(choices=row_choices, value=item["row_id"]),
            audio_path,
            item.get("row_id", ""),
            timestamp_label(item),
            highlighted_context(item),
            item.get("bronze_ar_text", ""),
            item.get("bronze_en_text", ""),
            item.get("selected_language", "ar-AR"),
            item.get("corrected_text", ""),
            progress,
            item.get("reviewer_id") or "",
            word_clip,
            segment_progress,
        )

    def jump_segment(label, language, corrected):
        save_current(language, corrected)
        return load_segment(labels.index(label))

    def previous_segment(language, corrected):
        save_current(language, corrected)
        return load_segment(state["segment_index"] - 1)

    def next_segment(language, corrected):
        save_current(language, corrected)
        return load_segment(state["segment_index"] + 1)

    def jump_row(row_id, language, corrected):
        save_current(language, corrected)
        row_ids = [x["row_id"] for x in state["items"]]
        if row_id in row_ids:
            state["row_index"] = row_ids.index(row_id)
        return render()

    def previous_row(language, corrected):
        save_current(language, corrected)
        state["row_index"] -= 1
        return render()

    def next_row(language, corrected):
        save_current(language, corrected)
        state["row_index"] += 1
        return render()

    def save(language, corrected):
        save_current(language, corrected)
        return render()

    def save_and_next(language, corrected):
        save_current(language, corrected)
        state["row_index"] += 1
        return render()

    def export_gold():
        label = labels[state["segment_index"]]
        out_json = gold_dir / f"{label}.json"
        out_jsonl = gold_dir / f"{label}.jsonl"
        doc = export_gold_annotations(
            state["items"],
            output_json_path=out_json,
            output_jsonl_path=out_jsonl,
            source_file=state["input_path"],
        )
        summary = doc["review_summary"]
        return (
            f"Exported:\n{out_json}\n{out_jsonl}\n\n"
            f"Reviewed rows: {summary['reviewed_rows']} / {summary['total_rows']}\n"
            f"Language changes: {summary['language_changed_rows']}\n"
            f"Text changes: {summary['text_changed_rows']}"
        )

    def load_table():
        return items_to_dataframe(state["items"]), f"Loaded {len(state['items'])} rows."

    def save_table(df):
        if df is None:
            return items_to_dataframe(state["items"]), "No table data received."

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        saved = 0
        previous_language = "ar-AR"
        errors = []

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
                    language = previous_language

                state["store"].save_review(
                    state["items"],
                    index,
                    selected_language=language,
                    corrected_text=corrected,
                )
                saved += 1
            except Exception as exc:
                errors.append(str(exc))

        msg = f"Bulk save complete. Updated rows: {saved}."
        if errors:
            msg += "\nErrors:\n" + "\n".join(errors[:10])

        return items_to_dataframe(state["items"]), msg

    def auto_detect_languages(df):
        if df is None:
            return items_to_dataframe(state["items"]), "No table data received."
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        df = df.copy()
        previous_language = "ar-AR"
        updated = 0

        for i, row in df.iterrows():
            corrected = str(row.get("Corrected Text", "") or "").strip()
            if corrected:
                detected = detect_language_from_script(corrected)
                previous_language = detected
            else:
                detected = previous_language

            if str(row.get("Language", "") or "") != detected:
                df.at[i, "Language"] = detected
                updated += 1

        return df, f"Auto-detected languages for {len(df)} rows. Updated language cells: {updated}."

    outputs = []

    with gr.Blocks(title="Manifest Segment Reviewer") as demo:
        gr.Markdown("# Manifest-driven Segment Reviewer")

        gr.HTML(
            """
            <script>
            document.addEventListener("keydown", function(e) {
              if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") {
                if (!(e.ctrlKey || e.metaKey)) return;
              }
              const click = (id) => {
                const el = document.querySelector(`#${id} button`);
                if (el) el.click();
              };
              if (e.ctrlKey && e.key === "s") { e.preventDefault(); click("save-btn"); }
              if (e.altKey && e.key === "ArrowRight") { e.preventDefault(); click("next-row-btn"); }
              if (e.altKey && e.key === "ArrowLeft") { e.preventDefault(); click("prev-row-btn"); }
              if (e.altKey && e.key === "ArrowDown") { e.preventDefault(); click("next-segment-btn"); }
              if (e.altKey && e.key === "ArrowUp") { e.preventDefault(); click("prev-segment-btn"); }
            });
            </script>
            <b>Shortcuts:</b> Ctrl/Cmd+S save · Alt+← previous row · Alt+→ next row · Alt+↑ previous segment · Alt+↓ next segment
            """
        )

        segment_progress = gr.Textbox(label="Segment Progress", interactive=False)

        segment_dropdown = gr.Dropdown(choices=labels, label="Segment", value=labels[0])

        jump_segment_btn = gr.Button("Load Segment", variant="primary")

        audio = gr.Audio(label="Segment Audio Player", interactive=False)

        with gr.Row():
            prev_segment_btn = gr.Button("Previous Segment", elem_id="prev-segment-btn")
            next_segment_btn = gr.Button("Next Segment", elem_id="next-segment-btn")

        with gr.Tab("Annotation Mode"):
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

        with gr.Tab("Review Mode"):
            row_dropdown = gr.Dropdown(label="Row Navigation")

            jump_row_btn = gr.Button("Jump to Row")

            row_id = gr.Textbox(label="Row ID", interactive=False)
            timestamps = gr.Textbox(label="Timestamps", interactive=False)
            context_html = gr.HTML(label="Current Word")

            word_audio = gr.Audio(label="Playback from Current Word", interactive=False)

            with gr.Row():
                prev_row_btn = gr.Button("Previous Row", elem_id="prev-row-btn")
                next_row_btn = gr.Button("Next Row", elem_id="next-row-btn")

            corrected_text = gr.Textbox(label="Corrected Text", lines=2)

            with gr.Row():
                bronze_ar = gr.Textbox(label="Bronze AR", interactive=False)
                bronze_en = gr.Textbox(label="Bronze EN", interactive=False)

            selected_language = gr.Radio(choices=LANGUAGE_CHOICES, label="Selected Language")
            progress = gr.Textbox(label="Progress", interactive=False)
            reviewer = gr.Textbox(label="Reviewer", interactive=False)

            with gr.Row():
                save_btn = gr.Button("Save", elem_id="save-btn")
                save_next_btn = gr.Button("Save & Next", variant="primary")
                export_btn = gr.Button("Export Gold")

            export_status = gr.Textbox(label="Export Status", interactive=False)

        outputs = [
            segment_dropdown,
            row_dropdown,
            audio,
            row_id,
            timestamps,
            context_html,
            bronze_ar,
            bronze_en,
            selected_language,
            corrected_text,
            progress,
            reviewer,
            word_audio,
            segment_progress,
        ]

        demo.load(fn=lambda: load_segment(0), outputs=outputs).then(
            fn=load_table,
            outputs=[annotation_table, table_status],
        )

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

        jump_segment_btn.click(
            fn=jump_segment,
            inputs=[segment_dropdown, selected_language, corrected_text],
            outputs=outputs,
        ).then(fn=load_table, outputs=[annotation_table, table_status])
        prev_segment_btn.click(
            fn=previous_segment,
            inputs=[selected_language, corrected_text],
            outputs=outputs,
        ).then(fn=load_table, outputs=[annotation_table, table_status])
        next_segment_btn.click(
            fn=next_segment,
            inputs=[selected_language, corrected_text],
            outputs=outputs,
        ).then(fn=load_table, outputs=[annotation_table, table_status])
        jump_row_btn.click(
            fn=jump_row,
            inputs=[row_dropdown, selected_language, corrected_text],
            outputs=outputs,
        )
        prev_row_btn.click(
            fn=previous_row,
            inputs=[selected_language, corrected_text],
            outputs=outputs,
        )
        next_row_btn.click(
            fn=next_row,
            inputs=[selected_language, corrected_text],
            outputs=outputs,
        )
        save_btn.click(fn=save, inputs=[selected_language, corrected_text], outputs=outputs)
        save_next_btn.click(fn=save_and_next, inputs=[selected_language, corrected_text], outputs=outputs)
        export_btn.click(fn=export_gold, outputs=export_status)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch manifest-driven Gradio segment reviewer.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reconciliation-dir", required=True)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(
        manifest_path=args.manifest,
        reconciliation_dir=args.reconciliation_dir,
    )
    allowed_paths = sorted({
        str(
            Path(row["audio_filepath"])
            .expanduser()
            .resolve()
            .parent
        )
        for row in load_manifest(args.manifest)
        if row.get("audio_filepath")
    })

    print("Allowed audio directories:")
    for path in allowed_paths:
        print(" -", path)

    app.launch(
        share=args.share,
        debug=True,
        allowed_paths=allowed_paths,
    )


if __name__ == "__main__":
    main()
