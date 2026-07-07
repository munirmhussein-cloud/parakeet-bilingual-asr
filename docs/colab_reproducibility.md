# Colab Reproducibility Guide

This repo is designed so a fresh Colab runtime can be restored quickly.

## Fresh Colab Startup

```python
from getpass import getpass
import subprocess
import os
from pathlib import Path

github_user = "munirmhussein-cloud"
repo_name = "parakeet-bilingual-asr"
token = getpass("GitHub PAT: ")
repo_url = f"https://{github_user}:{token}@github.com/{github_user}/{repo_name}.git"

if Path(repo_name).exists():
    subprocess.run(["rm", "-rf", repo_name], check=True)

subprocess.run(["git", "clone", repo_url], check=True)
os.chdir(repo_name)
print("Repo ready:", Path.cwd())
```

## Setup Environment

```bash
python scripts/setup_colab.py
```

Restart runtime if required.

## Verify Environment

```bash
cd /content/parakeet-bilingual-asr
python scripts/verify_environment.py
```

## Mount Google Drive and Check Artifacts

```bash
python scripts/mount_drive_and_check_artifacts.py
```

## Expected Result

- Repository clones
- Dependencies install
- GPU/CUDA detected
- NeMo imports
- Google Drive artifacts visible

## Important

Colab runtimes are ephemeral. Dependencies must be reinstalled in each new runtime, but the commands are now centralized in this repo.
