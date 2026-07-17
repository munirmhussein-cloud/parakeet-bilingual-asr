# HANDOFF — SILVER+ V3 FIFTH-VIEW INTEGRATION

Owner: WrittenASR. Implementer: Codex/producer pipeline. Reviewer: advisor thread.

## Ratified design

- The repaired Silver v3 multiview token lattice is the engine spine.
- Azure/Pyannote forced-Arabic enters as the lowest-priority fifth view: `azure_pyannote_forced_ar`.
- Azure votes on tashkeel-stripped, lightly folded Arabic skeleton identity only.
- When Azure corroborates a Parakeet lexeme, the Parakeet vocalized surface remains the emitted surface.
- When Azure provides Arabic script against a Latin/romanized Parakeet slot, Arabic script is preferred and `script_disagreement` is recorded.
- Azure-only tokens are Tier-C with `azure_only` and `needs_vocalization`; they are not treated as final vocalized text.
- Silver+ may add or corroborate but may never subtract Arabic letters relative to repaired Silver v3.

## Hard gates

- `azure_used_inside_silver_v3_lattice=true`
- per-segment Arabic-letter no-regression
- Arabic tashkeel density at least 0.20
- pilot segments 0–29: Arabic recall at least 0.33 and English recall at least 0.80 against Gold v12
- inherited immediate duplicate budget
- all five views present
- deterministic `seg_XXXXXX` mapping
- repository commit and configuration hash stamped into reports and packages

## Implementation

Primary executable:

```text
scripts/integrate_silver_plus_v3_fifth_view.py
```

The executable consumes repaired Silver v3 segment JSONL, the position-aligned Azure/Pyannote parent JSONL, and Gold-v12 segment JSONL. It emits integrated segment JSONL, token provenance JSONL, transcript JSON, quality report, and a stamped ZIP package with `PACKAGE_MANIFEST.json`.
