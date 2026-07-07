# Sprint 1 — Endpoint Capability Confirmation

## Confirmed endpoint

- Server: `grpc.nvcf.nvidia.com:443`
- Function ID: `71203149-d3b7-4460-8231-1be2543a1fca`
- Client: `nvidia-riva-client`
- Service method used: `offline_recognize`

## Confirmed RecognitionConfig fields

- encoding
- sample_rate_hertz
- language_code
- max_alternatives
- profanity_filter
- speech_contexts
- audio_channel_count
- enable_word_time_offsets
- enable_automatic_punctuation
- enable_separate_recognition_per_channel
- model
- verbatim_transcripts
- diarization_config
- custom_configuration
- endpointing_config

## Live probe findings

- `language_code="ar-AR"` works.
- `language_code="en-US"` did not force English on Arabic speech.
- `max_alternatives=3` was accepted, but only one alternative appeared in the tested output.
- Word timestamps are returned.
- Timestamp values appear to be integer milliseconds.
- Output is divided into result-level segments of roughly 8 seconds.
- Alternative-level confidence is returned.
- Per-word confidence was not visible.
- Token timestamps were not visible.
- Per-word language labels were not visible.
- Punctuation setting did not visibly affect the Arabic-only probe.

## Bronze schema implications

Bronze v1 should preserve:

- raw response per chunk
- request config per chunk
- result-level segments
- alternative-level confidence
- word timestamps in raw milliseconds
- converted local/global seconds
- placeholders for future token spans, language spans, and reconciliation flags
