#!/usr/bin/env python3
"""Run Copper v1: whole-lecture Faster-Whisper Turbo plus canonical 20s export."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lecture-id", required=True)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--download-root", type=Path, default=None)
    parser.add_argument("--reuse-whole-json", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio = args.audio.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    whole_json = output_dir / f"{args.lecture_id}_copper_v1_faster_whisper_turbo_whole.json"
    v23_runner = Path(__file__).with_name("run_bronze_v2_3_faster_whisper.py")
    exporter = Path(__file__).with_name("export_copper_v1_canonical.py")

    if not args.reuse_whole_json or not whole_json.is_file():
        command = [
            sys.executable,
            str(v23_runner),
            "--lecture-id",
            args.lecture_id,
            "--audio",
            str(audio),
            "--output",
            str(whole_json),
            "--model",
            args.model,
            "--device",
            "cuda",
            "--compute-type",
            args.compute_type,
            "--beam-size",
            str(args.beam_size),
            "--best-of",
            str(args.best_of),
            "--temperature",
            str(args.temperature),
            "--num-workers",
            "1",
            "--overwrite",
        ]
        if args.download_root is not None:
            command.extend([
                "--download-root",
                str(args.download_root.expanduser().resolve()),
            ])
        subprocess.run(command, check=True)

    export_command = [
        sys.executable,
        str(exporter),
        "--lecture-id",
        args.lecture_id,
        "--audio",
        str(audio),
        "--whole-json",
        str(whole_json),
        "--output-dir",
        str(output_dir),
        "--overwrite",
    ]
    subprocess.run(export_command, check=True)

    print("\n=== COPPER V1 COMPLETE ===")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
