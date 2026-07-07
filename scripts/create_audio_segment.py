"""
Create a normalized WAV segment for Sprint validation.

Input audio can be MP3/WAV/etc. Output is 16 kHz mono WAV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pydub import AudioSegment


def parse_args():
    parser = argparse.ArgumentParser(description="Create a normalized WAV segment.")
    parser.add_argument("--input", required=True, help="Input audio path.")
    parser.add_argument("--output", required=True, help="Output WAV segment path.")
    parser.add_argument("--start-ms", type=int, default=0)
    parser.add_argument("--duration-ms", type=int, default=10_000)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(args.sample_rate)

    end_ms = min(args.start_ms + args.duration_ms, len(audio))
    segment = audio[args.start_ms:end_ms]
    segment.export(output_path, format="wav")

    print({
        "input": str(input_path),
        "output": str(output_path),
        "start_ms": args.start_ms,
        "duration_seconds": round(len(segment) / 1000, 2),
        "sample_rate": args.sample_rate,
        "channels": 1,
    })


if __name__ == "__main__":
    main()
