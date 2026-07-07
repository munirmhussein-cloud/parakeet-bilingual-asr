import subprocess
import sys

print("=== Parakeet Bilingual ASR Colab Quickstart ===")
print("This script assumes the repo has already been cloned.")

steps = [
    [sys.executable, "scripts/setup_colab.py"],
    [sys.executable, "scripts/verify_environment.py"],
]

for step in steps:
    print(f"\n$ {' '.join(step)}")
    subprocess.run(step, check=True)

print("\n✅ Quickstart complete.")
