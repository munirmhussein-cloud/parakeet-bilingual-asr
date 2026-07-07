# Sprint 2 Minimal Gradio Annotator

This sprint adds a lightweight human-review scaffold for bilingual ASR transcripts.

## Added Components

- `apps/gradio_annotator.py` provides a minimal Gradio UI for reviewing one transcript segment and saving feedback.
- `src/annotation/gradio_data_model.py` defines transcript segment and annotation record objects.
- `src/annotation/review_store.py` stores in-progress reviews as JSON Lines under `annotations/progress/`.
- `src/annotation/gold_export.py` exports reviewed records into a gold annotation JSON document.
- `schemas/gold_annotations_v1.schema.json` documents the expected gold export shape.
- `examples/gold_annotations_v1_sample.json` provides a concrete sample document.

## Run the Annotator

```bash
python scripts/run_gradio_annotator.py
```

## Export Gold Annotations

```bash
python scripts/export_gold_annotations.py
```

The default export path is `annotations/gold/gold_annotations_v1.json`.

## Storage Policy

Progress and gold annotation folders are tracked with `.gitkeep` placeholders. Generated review and gold files should be reviewed before committing because they may contain transcript text from private audio.
