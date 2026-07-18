# Copper v1 Arabic Vocalization Experiment

Copper v1a through v1e are controlled Faster-Whisper Turbo profiles for evaluating Arabic lexical and vocalization fidelity on the same whole-lecture audio.

All profiles use:

- model: `turbo`
- device: CUDA
- compute type: `float16`
- one whole-lecture decode
- word timestamps when available
- deterministic contiguous canonical 20-second projection
- JSON, JSONL, DOCX, and package-manifest exports
- no VAD, preserving the same whole-audio input

## Profiles

| Profile | Purpose | Parameter change |
|---|---|---|
| Copper v1a | Control | Current proven Copper baseline |
| Copper v1b | Language switching | Per-segment multilingual detection; three initial detection segments |
| Copper v1c | Prompt bias | v1b plus a restrained bilingual Arabic terminology prompt |
| Copper v1d | Lexical bias | v1b plus Arabic and Islamic terminology hotwords |
| Copper v1e | Quality candidate | v1b plus prompt, hotwords, beam 8, patience 1.2, and temperature fallback 0.0/0.2/0.4 |

## Evaluation principles

Do not score Arabic character count alone. Compare each output against the reviewed reference and source audio for:

1. Correct Arabic lexical content.
2. Correct Arabic-script selection at English/Arabic switches.
3. Preservation of audible diacritics and vocalization.
4. Incorrect or invented diacritics.
5. Names, Qur'anic quotations, invocations, and honorific phrases.
6. Hallucinated Arabic during English speech.
7. Missing Arabic represented as English transliteration.
8. Repetition, dropped speech, and timestamp continuity.
9. Runtime and real-time factor.

The reviewed document is a comparison reference, not an assumption that every existing Arabic character or diacritic is correct. Audio remains the final authority for vocalization fidelity.
