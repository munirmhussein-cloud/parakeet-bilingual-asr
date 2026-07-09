#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def export_manifest(metadata_path, manifest_out):
    rows = read_jsonl(metadata_path)

    manifest_out = Path(manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    with manifest_out.open("w", encoding="utf-8") as f:
        for row in rows:
            manifest_row = {
                "audio_filepath": row["segment_audio_filepath"],
                "duration": row["duration"],
                "text": "",
                "source_audio_id": row["source_audio_id"],
                "segment_id": row["segment_id"],
                "start_time": row["start_time"],
                "end_time": row["end_time"]
            }
            f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

    print(f"Exported {len(rows)} segment manifest rows to {manifest_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--manifest-out", required=True)
    args = parser.parse_args()

    export_manifest(args.metadata, args.manifest_out)


if __name__ == "__main__":
    main()
