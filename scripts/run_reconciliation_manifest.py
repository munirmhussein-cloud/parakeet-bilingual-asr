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
    parser.add_argument("--bronze-ar-dir", required=True)
    parser.add_argument("--bronze-en-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[:args.limit]

    bronze_ar_dir = Path(args.bronze_ar_dir)
    bronze_en_dir = Path(args.bronze_en_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    failed = []
    total_items = 0

    for row in rows:
        segment_id = row["segment_id"]
        audio_path = row["audio_filepath"]
        safe_segment_id = safe_name(segment_id)

        bronze_ar = bronze_ar_dir / f"{safe_segment_id}.json"
        bronze_en = bronze_en_dir / f"{safe_segment_id}.json"
        output = output_dir / f"{safe_segment_id}_reconciliation.json"

        if not bronze_ar.exists() or not bronze_en.exists():
            failed.append({
                "segment_id": segment_id,
                "error": "missing_bronze_file",
                "bronze_ar_exists": bronze_ar.exists(),
                "bronze_en_exists": bronze_en.exists(),
            })
            continue

        cmd = [
            "python",
            "scripts/generate_reconciliation_gradio_input.py",
            "--bronze-ar", str(bronze_ar),
            "--bronze-en", str(bronze_en),
            "--output", str(output),
            "--audio-id", segment_id,
            "--audio-path", audio_path,
        ]

        print("\nRunning:")
        print(" ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
            completed += 1

            with output.open("r", encoding="utf-8") as f:
                doc = json.load(f)
            total_items += len(doc.get("items", []))

        except subprocess.CalledProcessError as exc:
            failed.append({
                "segment_id": segment_id,
                "error": "reconciliation_failed",
                "returncode": exc.returncode,
            })

    summary = {
        "manifest": args.manifest,
        "bronze_ar_dir": str(bronze_ar_dir),
        "bronze_en_dir": str(bronze_en_dir),
        "output_dir": str(output_dir),
        "attempted": len(rows),
        "completed": completed,
        "failed": len(failed),
        "total_items": total_items,
        "failures": failed,
    }

    summary_path = output_dir / "_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
