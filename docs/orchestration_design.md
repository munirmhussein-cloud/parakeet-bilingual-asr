# Orchestration Design

Sprint 3 recommends a single orchestration entry point while preserving modular scripts.

## Proposed Command

python run_pipeline.py \
  --audio data/audio/sample.wav \
  --run-bronze \
  --reconcile \
  --validate \
  --launch-gradio

## Principles

- Each underlying script remains independently runnable.
- The orchestration script should call existing modules or subprocesses.
- Validation should happen before Gradio launch when possible.
- Outputs should be deterministic and stored in predictable directories.
- Annotation and Gold export must remain separate from inference.

## Suggested Pipeline Stages

1. Environment verification
2. Audio preparation
3. Bronze EN inference
4. Bronze AR inference
5. Reconciliation generation
6. Gradio-ready input generation
7. Gold annotation validation
8. Gradio launch
9. Gold to NeMo manifest export
10. Gold to audit TSV export
