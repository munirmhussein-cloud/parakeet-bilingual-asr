\
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from pipeline.common.azure_fast import (
    API_VERSION,
    transcribe_file,
)
from pipeline.common.json_io import (
    atomic_write_json,
    read_json,
)


SCHEMA_VERSION = (
    "bronze_v3_azure_fast_whole_raw_v1"
)


def is_complete(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        payload = read_json(path)
    except Exception:
        return False

    return (
        payload.get("schema_version")
        == SCHEMA_VERSION
        and payload.get(
            "inference",
            {},
        ).get("status")
        == "completed"
        and isinstance(
            payload.get(
                "response"
            ),
            dict,
        )
    )


def summarize_response(
    response: dict[str, Any],
) -> dict[str, Any]:
    phrases = (
        response.get("phrases")
        or []
    )

    combined = (
        response.get(
            "combinedPhrases"
        )
        or []
    )

    word_count = 0
    locale_distribution = {}

    for phrase in phrases:
        words = (
            phrase.get("words")
            or []
        )

        word_count += len(words)

        locale = (
            phrase.get("locale")
            or "unknown"
        )

        locale_distribution[
            locale
        ] = (
            locale_distribution.get(
                locale,
                0,
            )
            + 1
        )

    combined_text = " ".join(
        str(item.get("text", "")).strip()
        for item in combined
        if str(
            item.get("text", "")
        ).strip()
    )

    return {
        "duration_milliseconds":
        response.get(
            "durationMilliseconds"
        ),
        "phrase_count":
        len(phrases),
        "combined_phrase_count":
        len(combined),
        "word_count":
        word_count,
        "combined_text_length":
        len(combined_text),
        "locale_distribution":
        locale_distribution,
        "has_text":
        bool(combined_text),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Azure Fast Transcription "
            "over one whole lecture."
        )
    )

    parser.add_argument(
        "--lecture-id",
        required=True,
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--locales",
        nargs="+",
        default=[
            "en-US",
            "ar-SA",
        ],
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if (
        is_complete(args.output)
        and not args.force
    ):
        print(
            {
                "status":
                "skipped_existing",
                "output":
                str(args.output),
            }
        )

        return 0

    key = os.environ.get(
        "AZURE_SPEECH_KEY",
        "",
    )

    region = os.environ.get(
        "AZURE_SPEECH_REGION",
        "",
    )

    started = time.perf_counter()

    response, request_metadata = (
        transcribe_file(
            audio_path=args.audio,
            subscription_key=key,
            region=region,
            locales=args.locales,
        )
    )

    summary = summarize_response(
        response
    )

    payload = {
        "schema_version":
        SCHEMA_VERSION,
        "lecture_id":
        args.lecture_id,
        "audio_filepath":
        str(args.audio.resolve()),
        "engine":
        "azure-speech",
        "mode":
        "fast_transcription_rest",
        "api_version":
        API_VERSION,
        "configuration": {
            "locales":
            args.locales,
            "language_forced":
            False,
            "reconciliation":
            False,
        },
        "inference": {
            "status":
            "completed",
            "runtime_seconds":
            round(
                time.perf_counter()
                - started,
                3,
            ),
            **request_metadata,
        },
        "summary":
        summary,
        "response":
        response,
    }

    atomic_write_json(
        args.output,
        payload,
    )

    print(
        {
            "lecture_id":
            args.lecture_id,
            "status":
            "completed",
            "output":
            str(args.output),
            "runtime_seconds":
            payload["inference"][
                "runtime_seconds"
            ],
            **summary,
        }
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
