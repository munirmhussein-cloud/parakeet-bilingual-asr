# Requirements

## colab_asr.txt

Primary Sprint 0 / Sprint 1 Colab ASR environment.

Includes:

- NVIDIA NeMo ASR
- NVIDIA Riva Python client
- audio processing dependencies
- compatibility pins discovered during Sprint 0

## Known Dependency Decision

The environment keeps:

```text
pandas==2.2.3
```

even though google-colab expects pandas==2.2.2.

Reason:

- NeMo pulls pyannote packages that require pandas>=2.2.3
- ASR pipeline compatibility is prioritized over the minor Colab package warning

Gradio is not part of the ASR runtime and should be isolated later.
