from pathlib import Path

try:
    from google.colab import drive
except ImportError:
    raise RuntimeError("This script is intended to run in Google Colab.")

print("=== Mounting Google Drive ===")
drive.mount("/content/drive")

base = Path("/content/drive/MyDrive/parakeet_bilingual_asr/artifacts/sprint0")

print("\nArtifact base:", base)

if not base.exists():
    raise FileNotFoundError(f"Sprint 0 artifact directory not found: {base}")

print("\nAvailable Sprint 0 artifact folders:")
for path in sorted(base.iterdir()):
    if path.is_dir():
        print("-", path)

print("\nSample artifact files:")
files = list(base.rglob("*"))
shown = 0

for f in files:
    if f.is_file():
        print("-", f)
        shown += 1
        if shown >= 25:
            break

print("\n✅ Google Drive artifacts are accessible.")
