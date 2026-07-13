# Multiview ASR Segmentation Validation

## Context

The fixed 20-second Silver workflow was found to omit audible speech in
some segments. Lecture 001 Segment 000011 returned zero words repeatedly
despite containing valid, audible 16 kHz mono PCM audio.

## Validated experiment

The same target interval was tested with multiple inference windows:

- 20 seconds, no padding: 0 words
- 24 seconds, ±2 seconds context: recovered the target speech
- 30 seconds, ±5 seconds context: partial and unstable recovery
- 40 seconds, ±10 seconds context: recovered the target speech
- continuous 5 minute 45 second inference: recovered the missing passage

The failure was deterministic for the isolated 20-second segment and was
not caused by reconciliation, Silver export, audio corruption, or random
endpoint instability.

## Production direction

Use separate inference and training units.

Routine inference views:

1. Whole lecture or long contiguous chunks
2. Existing fixed 20-second segments
3. Target segments inferred with ±2 seconds of contextual padding

Conditional recovery:

4. ±10-second contextual inference for empty, truncated, low-density,
   unexpected-script, or high-disagreement segments

The language argument is currently retained for API compatibility, but
the tested en-US and ar-AR requests produced effectively identical
multilingual hypotheses. Duplicate forced-language passes should not be
run unless future parity testing shows a material difference.

## Reconciliation requirements

All hypotheses must be mapped to lecture-global timestamps.

The canonical transcript must:

- align overlapping hypotheses rather than concatenate them;
- preserve one-sided words;
- retain source provenance;
- detect repeated phrases introduced by overlapping windows;
- use contextual inference to repair fixed-window omissions;
- assign canonical words to non-overlapping training segments using
  deterministic timestamp ownership;
- flag unresolved disagreement for review.

## Training-data rule

Overlapping and whole-audio inference views are evidence only.

The final NeMo Gold dataset must contain one canonical transcript for
each non-overlapping training audio interval. Overlapping inference
windows must not become duplicate training examples by default.

## Benchmark policy

Existing Bronze and Silver artifacts remain frozen as the baseline.
They do not need to be regenerated before the Lecture 001 NeMo pilot.
