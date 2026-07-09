import argparse
import json
from collections import defaultdict
from pathlib import Path


MILLISECONDS_THRESHOLD = 1000.0


def load_gold_document(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        items = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return {"items": items}

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"items": data}
    return {"items": [data]}


def get_items(document):
    if isinstance(document, dict) and "items" in document:
        return document["items"]
    if isinstance(document, list):
        return document
    return [document]


def clean_join(tokens):
    text = " ".join(t.strip() for t in tokens if str(t or "").strip())
    for punct in [".", ",", "?", "!", ":", ";"]:
        text = text.replace(f" {punct}", punct)
    return text.strip()


def maybe_ms_to_seconds(value):
    if value is None:
        return None
    value = float(value)
    if value > MILLISECONDS_THRESHOLD:
        return round(value / 1000.0, 3)
    return round(value, 3)


def resolve_audio_filepath(first, document, fallback):
    return (
        first.get("audio_filepath")
        or first.get("audio_path")
        or document.get("source_audio")
        or document.get("source_audio_filepath")
        or fallback
    )


def infer_duration_seconds(group, cli_duration):
    if cli_duration is not None:
        return round(float(cli_duration), 3)

    first = group[0]

    if first.get("duration") is not None:
        return maybe_ms_to_seconds(first.get("duration"))

    local_starts = [
        x.get("local_start")
        for x in group
        if isinstance(x.get("local_start"), (int, float))
    ]
    local_ends = [
        x.get("local_end")
        for x in group
        if isinstance(x.get("local_end"), (int, float))
    ]

    if local_starts and local_ends:
        return maybe_ms_to_seconds(max(local_ends) - min(local_starts))

    global_starts = [
        x.get("global_start")
        for x in group
        if isinstance(x.get("global_start"), (int, float))
    ]
    global_ends = [
        x.get("global_end")
        for x in group
        if isinstance(x.get("global_end"), (int, float))
    ]

    if global_starts and global_ends:
        return maybe_ms_to_seconds(max(global_ends) - min(global_starts))

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Export Gold annotations to NeMo ASR manifest JSONL."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio-filepath", default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--offset", type=float, default=None)
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Base path used to verify relative audio_filepaths.",
    )
    args = parser.parse_args()

    document = load_gold_document(args.input)
    items = get_items(document)

    grouped = defaultdict(list)
    for item in items:
        if not args.include_unreviewed and item.get("review_status") != "reviewed":
            continue

        audio_id = item.get("audio_id") or item.get("segment_id") or "unknown_audio"
        grouped[audio_id].append(item)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(args.repo_root)
    rows = []
    errors = []

    for audio_id, group in grouped.items():
        group = sorted(group, key=lambda x: x.get("word_index", x.get("index", 0)))

        text = clean_join([x.get("corrected_text", "") for x in group])
        first = group[0]

        audio_filepath = resolve_audio_filepath(first, document, args.audio_filepath)
        if not audio_filepath:
            errors.append(f"{audio_id}: missing audio_filepath")
            continue

        audio_path = Path(audio_filepath)
        check_path = audio_path if audio_path.is_absolute() else repo_root / audio_path
        if not check_path.exists():
            errors.append(f"{audio_id}: audio file does not exist: {audio_filepath}")
            continue

        duration = infer_duration_seconds(group, args.duration)
        if duration is None or duration <= 0:
            errors.append(f"{audio_id}: missing or invalid duration")
            continue

        if not text.strip():
            errors.append(f"{audio_id}: empty transcript text")
            continue

        row = {
            "audio_filepath": audio_filepath,
            "duration": duration,
            "text": text,
        }

        if args.offset is not None:
            row["offset"] = maybe_ms_to_seconds(args.offset)

        rows.append(row)

    if errors:
        raise ValueError("NeMo export validation failed:\n" + "\n".join(errors))

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input": args.input,
                "output": str(output_path),
                "manifest_rows": len(rows),
                "audio_ids": list(grouped.keys()),
                "duration_unit": "seconds",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
