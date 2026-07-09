"""
Run Bronze inference for one WAV file using the Sprint 1 Riva endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.parakeet_infer import (
    DEFAULT_RIVA_FUNCTION_ID,
    DEFAULT_RIVA_SERVER,
    transcribe_file,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Bronze ASR inference.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--audio-id", default=None)

    parser.add_argument("--model", default=None)
    parser.add_argument("--server", default=DEFAULT_RIVA_SERVER)
    parser.add_argument("--function-id", default=DEFAULT_RIVA_FUNCTION_ID)
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--no-ssl", action="store_true")
    parser.add_argument("--no-automatic-punctuation", action="store_true")
    parser.add_argument("--no-verbatim-transcripts", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")

    bronze = transcribe_file(
        args.audio,
        model_name=args.model,
        language=args.language,
        audio_id=args.audio_id,
        server=args.server,
        function_id=args.function_id,
        api_key=api_key,
        use_ssl=not args.no_ssl,
        automatic_punctuation=not args.no_automatic_punctuation,
        verbatim_transcripts=not args.no_verbatim_transcripts,
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
                "backend": bronze["backend"],
                "language": bronze["language"],
                "word_count": len(bronze["words"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
