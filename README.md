# Bilingual ASR Parakeet

Project scaffold for bilingual automatic speech recognition experiments using NVIDIA Parakeet/NeMo tooling.

## Repository layout

- `notebooks/` — Colab and local notebooks for environment validation and experiments.
- `src/audio/` — Audio metadata utilities.
- `src/inference/` — Parakeet inference entry points.
- `src/export/` — Transcript export helpers.
- `configs/` — Model and training configuration files.
- `data/` — Local data placeholders; raw audio and generated data stay out of git.
- `experiments/` — Experiment notes and run-specific assets.
- `artifacts/` — Generated outputs and model artifacts; only placeholders are tracked.
- `docs/` — Environment, model selection, and storage documentation.
- `requirements/` — Python dependency specifications.
