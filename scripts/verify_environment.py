import subprocess
import sys
import shutil
import importlib.metadata as metadata

print("=== Python ===")
print(sys.version)

print("\n=== GPU ===")
nvidia_smi = shutil.which("nvidia-smi")

if nvidia_smi:
    subprocess.run([nvidia_smi], check=False)
else:
    print("nvidia-smi not found on PATH")

print("\n=== Package versions ===")
packages = [
    "nemo-toolkit",
    "torch",
    "torchaudio",
    "numpy",
    "pandas",
    "packaging",
    "requests",
    "fsspec",
    "huggingface-hub",
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

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: None")

print("nemo import: ✅")
print("nemo.collections.asr import: ✅")
print("soundfile import: ✅")
print("✅ Environment verified.")
