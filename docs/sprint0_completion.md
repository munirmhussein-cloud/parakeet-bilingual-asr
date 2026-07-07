# Sprint 0 Completion Report

## Project

Bilingual English/Arabic ASR dataset pipeline using NVIDIA Parakeet multilingual ASR, NVIDIA NeMo, Google Colab, Google Drive, and GitHub.

## Sprint 0 Objective

Establish a reproducible Colab environment capable of:

- Running NVIDIA NeMo
- Uploading and validating audio
- Normalizing audio to 16 kHz mono WAV
- Running ASR inference
- Accessing NVIDIA hosted Parakeet multilingual ASR
- Exporting transcript outputs
- Preserving word-level timestamps
- Persisting artifacts to Google Drive
- Reconstructing the environment from GitHub

## Environment Validated

- Runtime: Google Colab
- Python: 3.12.13
- GPU used: NVIDIA L4
- CUDA visible to PyTorch: Yes
- A100 requirement: Deferred until training-scale work

## Key Packages

- nemo-toolkit: 2.7.3
- torch: 2.11.0+cu128
- torchaudio: 2.11.0+cu128
- nvidia-riva-client: 2.16.0
- pandas: 2.2.3
- numpy: 2.0.2
- fsspec: 2024.12.0
- huggingface-hub: 0.36.2

## Parakeet Access

Model tested through NVIDIA Build / hosted Riva gRPC endpoint.

- Server: grpc.nvcf.nvidia.com:443
- Function ID: 71203149-d3b7-4460-8231-1be2543a1fca
- Model: Parakeet 1.1B RNNT Multilingual ASR

## Major Finding

Hosted Parakeet supports:

- English transcription
- Arabic transcription
- Code-switched English/Arabic transcription
- Automatic punctuation
- Word-level timestamps

The initial plain transcript lacked punctuation/timestamps only because the sample client did not request them.

## Audio Tested

Input audio:

- Seerah – 01 Specialities of Prophet Muhammed Part 1.mp3
- Duration: ~45.74 minutes
- Original format: MP3, 44.1 kHz, stereo
- Normalized format: WAV, 16 kHz, mono

Test clip:

- 60 seconds
- 16 kHz
- mono
- successfully transcribed by hosted Parakeet

## Artifact Storage

Large audio and generated artifacts are stored in Google Drive, not GitHub.

Google Drive base path:

```text
/content/drive/MyDrive/parakeet_bilingual_asr/artifacts/sprint0/
```

Latest Sprint 0 backup observed:

```text
/content/drive/MyDrive/parakeet_bilingual_asr/artifacts/sprint0/sprint0_artifacts_20260707_051311
```

## GitHub Role

GitHub stores:

- setup scripts
- verification scripts
- docs
- schemas
- notebooks
- lightweight configs

GitHub does not store:

- audio
- WAV files
- MP3 files
- generated transcript artifacts
- API keys
- model checkpoints

## Sprint 0 Status

Sprint 0 is complete.
