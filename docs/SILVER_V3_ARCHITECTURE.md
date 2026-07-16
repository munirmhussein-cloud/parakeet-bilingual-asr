# Silver v3 Architecture

## Purpose

Silver v3 produces a high-recall automatic transcript from multiple NVIDIA
Parakeet inference views while retaining canonical 20-second output segments.

Silver v3 does not use Azure or human-reviewed Gold data.

## Input views

1. Whole lecture
   - Provides global lexical and chronological evidence.

2. Canonical 20-second segments
   - Defines authoritative output boundaries.
   - Provides locally decoded transcript evidence.

3. 10-second windows with 5-second stride
   - Provides overlapping phrase and boundary evidence.
   - Self-overlap must be collapsed through alignment before multiview voting.

4. Contiguous 2.5-second windows
   - Provides local omission evidence.
   - Token timestamps may be treated as window-level evidence when unreliable.

## Reconciliation

Silver v3 uses time-keyed align-then-vote reconciliation.

It must never:

- concatenate complete transcript hypotheses;
- choose a single transcript source for the whole segment;
- silently discard an evidence observation;
- treat repeated observation as newly spoken content.

Each probable spoken-token position becomes one lattice slot.

Every emitted token retains:

- selected surface form;
- normalized form;
- approximate time;
- acceptance tier;
- contributing views;
- source observations;
- alternative hypotheses;
- single-witness status.

## Acceptance tiers

- Tier A: supported by at least two independent views.
- Tier B: supported by multiple overlapping context windows.
- Tier C: observed by one source only and retained with a review flag.

## Output

Silver v3 exports:

- canonical 20-second segment JSONL;
- full JSON;
- token-provenance JSONL;
- DOCX;
- validation report;
- packaged ZIP.

## Required validation

- canonical segment order preserved;
- unique segment IDs;
- no chronology regression;
- zero unaccounted evidence observations;
- zero immediate duplicated six-grams;
- deterministic serialized JSON/JSONL.
