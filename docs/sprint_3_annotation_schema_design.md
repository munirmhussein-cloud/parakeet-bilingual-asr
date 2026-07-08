# Sprint 3 — Annotation Schema Design

Sprint 3 defines the production Gold annotation format for bilingual English/Arabic ASR correction.

## Goals

- Preserve Bronze EN and Bronze AR outputs.
- Capture corrected Gold text.
- Support token-level and span-level review.
- Track selected language per token.
- Preserve audit history.
- Export clean NeMo-compatible ASR manifests.

## Gold Annotation v1

Gold annotations are stored as rich JSON or JSONL records.

Each record represents one audio segment and contains:

- audio path
- optional offset
- duration
- final corrected text
- normalized text
- primary language
- code-switching flag
- reviewer metadata
- token-level annotations
- audit history

## Language Labels

Allowed token labels:

- `eng`
- `ar`
- `mixed`
- `noise`
- `unintelligible`
- `silence`

## NeMo Export

Training export should remain minimal.

Each NeMo manifest row should contain:

```json
{"audio_filepath":"path.wav","duration":4.2,"text":"corrected transcript"}
