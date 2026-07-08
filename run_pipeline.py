import argparse
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def run(cmd, env):
    print("\n$", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run bilingual ASR annotation workflow.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--audio-id", default="live_validation_segment_001")
    parser.add_argument("--run-bronze", action="store_true")
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--launch-gradio", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"

    audio = Path(args.audio)
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    Path("data/transcripts").mkdir(parents=True, exist_ok=True)
    Path("data/annotations").mkdir(parents=True, exist_ok=True)
    Path("data/exports").mkdir(parents=True, exist_ok=True)

    bronze_en = Path(f"data/transcripts/bronze_en_{args.audio_id}.json")
    bronze_ar = Path(f"data/transcripts/bronze_ar_{args.audio_id}.json")
    gradio_input = Path("data/annotations/gradio_reconciliation_input_v1.json")

    if args.run_bronze:
        if bronze_en.exists() and bronze_ar.exists() and not args.force:
            print("Bronze outputs already exist; skipping Bronze inference. Use --force to rerun.")
        else:
            run([
                "python", "scripts/run_bronze_inference.py",
                "--audio", str(audio),
                "--output", str(bronze_en),
                "--language", "en-US",
                "--audio-id", args.audio_id,
            ], env)

            run([
                "python", "scripts/run_bronze_inference.py",
                "--audio", str(audio),
                "--output", str(bronze_ar),
                "--language", "ar-AR",
                "--audio-id", args.audio_id,
            ], env)

    if args.reconcile:
        if gradio_input.exists() and not args.force:
            print("Gradio reconciliation input already exists; skipping reconciliation. Use --force to rerun.")
        else:
            run([
                "python", "scripts/generate_reconciliation_gradio_input.py",
                "--bronze-ar", str(bronze_ar),
                "--bronze-en", str(bronze_en),
                "--output", str(gradio_input),
                "--audio-id", args.audio_id,
                "--audio-path", str(audio),
            ], env)

    if args.validate:
        run([
            "python", "scripts/validate_gradio_input.py",
            "--input", str(gradio_input),
        ], env)

    if args.launch_gradio:
        run([
            "python", "scripts/run_gradio_annotator.py",
            "--input", str(gradio_input),
        ], env)


if __name__ == "__main__":
    main()
