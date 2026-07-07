# Sprint 1 — Scalable Manifests

Sprint 1 uses three scalable manifest levels:

1. `collection_manifest.json` tracks the full corpus target, currently planned for 105 audio hours.
2. `lecture_manifest.json` tracks one lecture, its normalized audio, chunking strategy, runs, and exports.
3. `run_manifest.json` tracks one ASR execution run, endpoint configuration, request settings, paths, and resume policy.
4. `chunk_manifest.json` tracks chunk-level state and will be populated during chunk generation.

Large generated manifests are persisted in Google Drive. The reusable generator script is stored in GitHub:

`python scripts/create_sprint1_manifests.py`
