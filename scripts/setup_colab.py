import subprocess
import sys


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


print("=== Installing system dependencies ===")
run(["apt-get", "update", "-qq"])
run(["apt-get", "install", "-y", "-qq", "libsndfile1", "ffmpeg", "sox"])

print("=== Installing Python dependencies ===")
run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "wheel", "Cython"])
run([sys.executable, "-m", "pip", "install", "-r", "requirements/colab_asr.txt"])

print("✅ Colab ASR setup complete.")
