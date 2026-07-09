#!/usr/bin/env python3

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(cmd):
    subprocess.run(cmd, check=True)


def ffprobe_duration(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def segment_audio(input_path, output_dir, metadata_out, segment_seconds=20, sample_rate=16000):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    metadata_out = Path(metadata_out)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)

    source_audio_id = input_path.stem
    total_duration = ffprobe_duration(input_path)

    pattern = output_dir / f"{source_audio_id}_seg_%06d.wav"

    run([
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "segment",
        "-segment_time", str(segment_seconds),
        "-reset_timestamps", "1",
        str(pattern),
    ])

    segment_files = sorted(output_dir.glob(f"{source_audio_id}_seg_*.wav"))

    rows = []
    created_at = datetime.now(timezone.utc).isoformat()

    for index, segment_path in enumerate(segment_files):
        segment_duration = ffprobe_duration(segment_path)
        start = round(index * segment_seconds, 3)
        end = round(min(start + segment_duration, total_duration), 3)

        rows.append({
            "source_audio_id": source_audio_id,
            "source_audio_filepath": str(input_path),
            "segment_id": segment_path.stem,
            "segment_audio_filepath": str(segment_path),
            "start_time": start,
            "end_time": end,
            "duration": round(segment_duration, 3),
            "sample_rate": sample_rate,
            "channels": 1,
            "segmentation_method": "ffmpeg_segment_muxer_fixed_window",
            "segmentation_version": "v1",
            "created_at": created_at,
            "validation_status": "pending"
        })

    with metadata_out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Generated {len(rows)} segments")
    print(f"Metadata written to {metadata_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata-out", required=True)
    parser.add_argument("--segment-seconds", type=float, default=20.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    segment_audio(
        args.input,
        args.output_dir,
        args.metadata_out,
        args.segment_seconds,
        args.sample_rate,
    )


if __name__ == "__main__":
    main()
