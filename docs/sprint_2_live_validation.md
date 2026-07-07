# Sprint 2 Live Validation Workflow

This workflow validates the full Sprint 0-2 path in a fresh Colab notebook:

```text
Upload audio
→ normalize WAV
→ create 10-second segment
→ live Riva Bronze AR
→ live Riva Bronze EN
→ generate Gradio reconciliation input
→ review in Gradio
→ save progress
→ resume
→ export Gold JSON / JSONL
