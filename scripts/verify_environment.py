import subprocess
import sys
import importlib.metadata as metadata

print("=== Python ===")
print(sys.version)

print("\n=== GPU ===")
subprocess.run(["nvidia-smi"], check=False)

print("\n=== Package versions ===")
packages = [
    "nemo-toolkit",
    "torch",
    "torchaudio",
    "numpy",
    "pandas",
    "nvidia-riva-client",
    "soundfile",
]
for pkg in packages:
    try:
        print(f"{pkg}: {metadata.version(pkg)}")
    except metadata.PackageNotFoundError:
        print(f"{pkg}: not installed")

print("\n=== Import validation ===")
import torch
import nemo
import nemo.collections.asr as nemo_asr
import soundfile

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("nemo import: ✅")
print("nemo.collections.asr import: ✅")
print("soundfile import: ✅")
print("✅ Environment verified.")
