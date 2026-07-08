# Gradio Annotation Mode Spec

The current word-by-word annotator becomes secondary Review Mode.

The new primary interface is Annotation Mode.

## Layout

| Index | Bronze EN | Bronze AR | Corrected Text | Language | Status |
|---|---|---|---|---|---|

## Required Behavior

- Load all token rows for the active segment.
- Allow inline editing of corrected text.
- Allow language selection per row.
- Save all edits in one bulk action.
- Preserve original Bronze EN and Bronze AR values.
- Append audit history on every changed field.
- Resume from the latest saved Gold annotation state.

## Save Behavior

On save:

1. Compare current values to previous values.
2. Update token fields.
3. Append audit events.
4. Regenerate segment-level `gold_text`.
5. Update `updated_at`.
6. Persist Gold JSON/JSONL.
