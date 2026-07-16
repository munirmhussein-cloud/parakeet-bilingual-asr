\
from __future__ import annotations

import json
import mimetypes
import time
from pathlib import Path
from typing import Any

import requests


API_VERSION = "2025-10-15"


def build_endpoint(region: str) -> str:
    region = region.strip()

    if not region:
        raise ValueError("Azure Speech region is empty.")

    return (
        f"https://{region}.api.cognitive.microsoft.com/"
        "speechtotext/transcriptions:transcribe"
    )


def transcribe_file(
    *,
    audio_path: Path,
    subscription_key: str,
    region: str,
    locales: list[str],
    timeout_seconds: int = 7200,
    retry_delays: tuple[float, ...] = (
        2.0,
        5.0,
        15.0,
    ),
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    if not subscription_key.strip():
        raise ValueError("Azure Speech key is empty.")

    endpoint = build_endpoint(region)

    definition = {
        "locales": locales,
    }

    headers = {
        "Ocp-Apim-Subscription-Key":
        subscription_key,
    }

    content_type = (
        mimetypes.guess_type(
            audio_path.name
        )[0]
        or "application/octet-stream"
    )

    attempts = len(retry_delays) + 1
    last_error: Exception | None = None

    for attempt_index in range(attempts):
        started = time.perf_counter()

        try:
            with audio_path.open("rb") as handle:
                response = requests.post(
                    endpoint,
                    params={
                        "api-version":
                        API_VERSION,
                    },
                    headers=headers,
                    files={
                        "audio": (
                            audio_path.name,
                            handle,
                            content_type,
                        ),
                    },
                    data={
                        "definition":
                        json.dumps(
                            definition,
                            ensure_ascii=False,
                        ),
                    },
                    timeout=timeout_seconds,
                )

            request_seconds = (
                time.perf_counter()
                - started
            )

            if response.status_code >= 400:
                raise RuntimeError(
                    "Azure Fast Transcription "
                    f"HTTP {response.status_code}: "
                    f"{response.text[:4000]}"
                )

            payload = response.json()

            metadata = {
                "endpoint": endpoint,
                "api_version": API_VERSION,
                "status_code":
                response.status_code,
                "request_seconds":
                round(request_seconds, 3),
                "attempt":
                attempt_index + 1,
                "locales":
                locales,
                "audio_size_bytes":
                audio_path.stat().st_size,
                "content_type":
                content_type,
            }

            return payload, metadata

        except Exception as error:
            last_error = error

            if attempt_index >= len(
                retry_delays
            ):
                break

            time.sleep(
                retry_delays[
                    attempt_index
                ]
            )

    raise RuntimeError(
        f"Azure transcription failed: "
        f"{last_error!r}"
    )
