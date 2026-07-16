# Lecture v3 Transcript Pipeline

## Stages

Raw lecture audio
    -> Bronze v3
    -> Silver v3
    -> Silver+ v3

## Bronze v3

Engine: Microsoft Azure Speech to Text

Mode: Fast Transcription REST API

Requested locales:

- en-US
- ar-SA

Bronze v3 preserves raw Azure output, phrase hypotheses, locale information,
word timing, confidence, operation metadata, failures and normalized JSONL.

Bronze v3 performs no reconciliation.

## Silver v3

Engine: NVIDIA Parakeet

Views:

- whole lecture;
- canonical 20-second segments;
- 10-second windows with 5-second stride;
- contiguous 2.5-second windows.

Reconciliation:

- align;
- collapse re-observations;
- vote by temporal token position;
- retain single-witness evidence with flags.

## Silver+ v3

- Pyannote decomposes each canonical 20-second parent.
- Azure forced ar-SA transcribes each Pyannote child.
- Child text is reconstructed chronologically.
- Populated Azure parent text is selected.
- Silver v3 is the fallback when Azure is empty.

## Required properties

- lecture-selectable;
- resumable;
- deterministic;
- no completed inference rerun unless force is explicitly supplied;
- JSON, JSONL, DOCX and ZIP exports;
- validation report for every stage.
