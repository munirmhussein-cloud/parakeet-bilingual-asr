# Silver+ v3 Architecture

## Purpose

Silver+ v3 combines the completed Silver v3 transcript with Azure forced-ar
transcription generated from Pyannote-defined speech clips.

Silver+ v3 deliberately preserves the successful Silver+ v2 construction.

## Pyannote and Azure sequence

For every canonical 20-second parent segment:

1. Run Pyannote within the parent audio.
2. Produce one or more speech-turn child clips.
3. Preserve each child's actual local and global start/end span.
4. Call Azure Speech to Text with forced locale ar-SA on each child clip.
5. Sort child transcripts by actual Pyannote global start time.
6. Reconstruct one Azure parent transcript.

Pyannote timing determines where and when Azure is called.

## Parent resolution

Silver+ v3 does not insert Azure tokens into the Silver v3 lattice.

Resolution is performed once per canonical parent:

- populated Azure parent transcript -> Silver+ v3 transcript;
- empty Azure parent transcript -> Silver v3 fallback.

## Parent and child timing

Parent boundaries always come from the authoritative canonical 20-second
manifest.

Pyannote child timing remains attached to the parent record for audit.

## Output

Silver+ v3 exports:

- canonical 20-second segment JSONL;
- full JSON;
- resolution-audit JSONL;
- DOCX;
- report;
- packaged ZIP.

Each record preserves:

- Silver v3 text;
- Azure parent text;
- selected Silver+ v3 text;
- resolution source;
- Pyannote child spans and transcripts;
- Azure completion statistics.

## Required validation

- segment IDs and order match Silver v3;
- parent spans exactly match the canonical manifest;
- Pyannote child chronology is valid;
- no token-level Azure/Silver merge occurred;
- every resolution source is explicit.
