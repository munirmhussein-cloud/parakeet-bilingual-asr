from pathlib import Path
import importlib.util
import os
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

os.chdir(REPO_ROOT)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIREMENTS = REPO_ROOT / "requirements.txt"

def package_available(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None

missing = []

checks = {
    "gradio": "gradio",
    "pandas": "pandas",
    "yaml": "pyyaml",
    "riva": "nvidia-riva-client",
    "jsonschema": "jsonschema",
}

for import_name, package_name in checks.items():
    if not package_available(import_name):
        missing.append(package_name)

if missing:
    print("Installing missing packages:", ", ".join(missing))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", *missing],
        check=True,
    )

print("Repo root:", REPO_ROOT)
print("cwd:", os.getcwd())
print("repo on sys.path:", str(REPO_ROOT) in sys.path)
print("NVIDIA_API_KEY set:", bool(os.environ.get("NVIDIA_API_KEY")))
print("Bootstrap complete.")
