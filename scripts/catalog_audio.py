#!/usr/bin/env python3

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe_metadata(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels:format=duration",
            "-of", "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    return {
        "duration": float(data["format"]["duration"]),
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
    }


def catalog_audio(input_path, catalog_out, source_audio_id=None, notes=""):
    input_path = Path(input_path)
    catalog_out = Path(catalog_out)
    catalog_out.parent.mkdir(parents=True, exist_ok=True)

    meta = ffprobe_metadata(input_path)

    row = {
        "source_audio_id": source_audio_id or input_path.stem,
        "source_audio_filepath": str(input_path),
        "sha256": sha256_file(input_path),
        "duration": round(meta["duration"], 3),
        "sample_rate": meta["sample_rate"],
        "channels": meta["channels"],
        "file_size_bytes": input_path.stat().st_size,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "catalog_version": "v1",
        "processing_status": "cataloged",
        "notes": notes,
    }

    with catalog_out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(row, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--catalog-out", required=True)
    parser.add_argument("--source-audio-id")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    catalog_audio(args.input, args.catalog_out, args.source_audio_id, args.notes)


if __name__ == "__main__":
    main()
