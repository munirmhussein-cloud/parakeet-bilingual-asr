# Storage Strategy

Large and sensitive files must not be committed to git.

- Store raw audio under `data/audio/` locally or in external object storage.
- Store generated manifests under `data/manifests/` when they are safe to share.
- Store transcript outputs under `data/transcripts/`.
- Store human annotations under `data/annotations/`.
- Store generated run artifacts under `artifacts/`.

The repository tracks `.gitkeep` placeholders so required directories exist after checkout.
