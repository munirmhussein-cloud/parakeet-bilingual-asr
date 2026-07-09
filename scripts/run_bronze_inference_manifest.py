#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_name(value):
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("–", "-")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    failed = []

    for row in rows:
        audio_path = row["audio_filepath"]
        segment_id = row["segment_id"]
        output_path = output_dir / f"{safe_name(segment_id)}.json"

        cmd = [
            "python",
            "scripts/run_bronze_inference.py",
            "--audio", audio_path,
            "--output", str(output_path),
            "--language", args.language,
            "--audio-id", segment_id,
        ]

        print("\nRunning:")
        print(" ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
            completed += 1
        except subprocess.CalledProcessError as exc:
            failed.append({
                "segment_id": segment_id,
                "audio_filepath": audio_path,
                "returncode": exc.returncode,
            })

    summary = {
        "language": args.language,
        "manifest": args.manifest,
        "output_dir": str(output_dir),
        "attempted": len(rows),
        "completed": completed,
        "failed": len(failed),
        "failures": failed,
    }

    summary_path = output_dir / "_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
