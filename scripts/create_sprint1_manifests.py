from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess


def sha256_file(path: Path):
    if not path or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ffprobe_duration(path: Path):
    if not path.exists():
        return None
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return None
    return float(r.stdout.strip())


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drive-root", required=True)
    p.add_argument("--lecture-id", required=True)
    p.add_argument("--lecture-title", required=True)
    p.add_argument("--normalized-wav", required=True)
    p.add_argument("--source-mp3", default=None)
    p.add_argument("--run-id", default=None)
    args = p.parse_args()

    now = datetime.now(timezone.utc).isoformat()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    drive_root = Path(args.drive_root)
    lecture_id = args.lecture_id
    normalized_wav = Path(args.normalized_wav)
    source_mp3 = Path(args.source_mp3) if args.source_mp3 else None

    collection_root = drive_root / "artifacts" / "sprint_1_bronze_transcript_generation"
    lectures_root = collection_root / "lectures"
    runs_root = collection_root / "runs"

    lecture_root = lectures_root / lecture_id
    run_root = runs_root / run_id

    duration_sec = ffprobe_duration(normalized_wav)

    collection_manifest = {
        "schema_version": "collection_manifest_v1",
        "project": "parakeet-bilingual-asr",
        "sprint": "sprint_1_bronze_transcript_generation",
        "created_at_utc": now,
        "updated_at_utc": now,
        "target_total_audio_hours": 105,
        "lecture_count": 1,
        "lectures": [
            {
                "lecture_id": lecture_id,
                "title": args.lecture_title,
                "duration_sec": duration_sec,
                "duration_hours": round(duration_sec / 3600, 4) if duration_sec else None,
                "status": "prepared",
                "lecture_manifest_path": str(lecture_root / "lecture_manifest.json")
            }
        ]
    }

    lecture_manifest = {
        "schema_version": "lecture_manifest_v1",
        "project": "parakeet-bilingual-asr",
        "sprint": "sprint_1_bronze_transcript_generation",
        "lecture_id": lecture_id,
        "title": args.lecture_title,
        "created_at_utc": now,
        "status": "prepared",
        "audio": {
            "source_mp3_path": str(source_mp3) if source_mp3 else None,
            "source_mp3_sha256": sha256_file(source_mp3) if source_mp3 else None,
            "normalized_wav_path": str(normalized_wav),
            "normalized_wav_sha256": sha256_file(normalized_wav),
            "duration_sec": duration_sec,
            "sample_rate_hz": 16000,
            "channels": 1,
            "format": "wav"
        },
        "chunking": {
            "strategy": "fixed_duration_with_overlap",
            "chunk_duration_sec": 300,
            "overlap_sec": 10,
            "chunk_manifest_path": str(lecture_root / "chunk_manifest.json")
        },
        "runs": [
            {
                "run_id": run_id,
                "run_manifest_path": str(run_root / "run_manifest.json"),
                "status": "initialized"
            }
        ],
        "outputs": {
            "bronze_json_path": None,
            "bronze_txt_path": None,
            "bronze_words_tsv_path": None,
            "bronze_segments_tsv_path": None
        }
    }

    run_manifest = {
        "schema_version": "run_manifest_v1",
        "project": "parakeet-bilingual-asr",
        "sprint": "sprint_1_bronze_transcript_generation",
        "run_id": run_id,
        "created_at_utc": now,
        "lecture_id": lecture_id,
        "lecture_title": args.lecture_title,
        "status": "initialized",
        "provider": "nvidia",
        "endpoint_server": "grpc.nvcf.nvidia.com:443",
        "function_id": "71203149-d3b7-4460-8231-1be2543a1fca",
        "schema_version_target": "bronze_transcript_v1",
        "asr_request": {
            "encoding": "LINEAR_PCM",
            "encoding_value": 1,
            "sample_rate_hertz": 16000,
            "language_code": "ar-AR",
            "max_alternatives": 3,
            "enable_word_time_offsets": True,
            "enable_automatic_punctuation": True,
            "verbatim_transcripts": False,
            "profanity_filter": False,
            "speech_contexts": [],
            "audio_channel_count": 1,
            "model": None,
            "custom_configuration": {},
            "endpointing_config": {}
        },
        "paths": {
            "lecture_root": str(lecture_root),
            "run_root": str(run_root),
            "raw_responses": str(run_root / "raw"),
            "parsed_chunks": str(run_root / "parsed"),
            "exports": str(run_root / "exports"),
            "logs": str(run_root / "logs")
        },
        "resume_policy": {
            "skip_completed_chunks": True,
            "save_raw_before_parse": True,
            "retry_failed_chunks_only": True
        }
    }

    chunk_manifest = {
        "schema_version": "chunk_manifest_v1",
        "project": "parakeet-bilingual-asr",
        "sprint": "sprint_1_bronze_transcript_generation",
        "lecture_id": lecture_id,
        "run_id": run_id,
        "created_at_utc": now,
        "chunk_duration_sec": 300,
        "overlap_sec": 10,
        "chunks": []
    }

    for d in [
        lecture_root,
        run_root / "raw",
        run_root / "parsed",
        run_root / "exports",
        run_root / "logs",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    write_json(collection_root / "collection_manifest.json", collection_manifest)
    write_json(lecture_root / "lecture_manifest.json", lecture_manifest)
    write_json(lecture_root / "chunk_manifest.json", chunk_manifest)
    write_json(run_root / "run_manifest.json", run_manifest)

    print("Collection manifest:", collection_root / "collection_manifest.json")
    print("Lecture manifest:", lecture_root / "lecture_manifest.json")
    print("Chunk manifest:", lecture_root / "chunk_manifest.json")
    print("Run manifest:", run_root / "run_manifest.json")


if __name__ == "__main__":
    main()
