"""
Run live Bronze inference for one WAV file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.inference.parakeet_infer import transcribe_file


def parse_args():
    parser = argparse.ArgumentParser(description="Run Bronze ASR inference.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default=None)
    parser.add_argument("--audio-id", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    bronze = transcribe_file(
        args.audio,
        model_name=args.model,
        language=args.language,
        audio_id=args.audio_id,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(bronze, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": str(output),
                "audio_id": bronze["audio_id"],
                "model_name": bronze["model_name"],
                "language": bronze["language"],
                "word_count": len(bronze["words"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
