import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_gold(path):
    path = Path(path)

    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "items" in data:
        return data["items"]

    if isinstance(data, list):
        return data

    return [data]


def clean_join(tokens):
    text = " ".join(t.strip() for t in tokens if str(t or "").strip())

    # light punctuation cleanup
    for punct in [".", ",", "?", "!", ":", ";"]:
        text = text.replace(f" {punct}", punct)

    return text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Export Gold annotations to NeMo ASR manifest JSONL."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--audio-filepath",
        default=None,
        help="Audio filepath to write into manifest rows when Gold items do not contain one.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration to write into manifest rows when Gold items do not contain one.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=None,
        help="Optional offset for segment slice manifests.",
    )
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Include unreviewed rows. Default exports only reviewed rows.",
    )
    args = parser.parse_args()

    items = load_gold(args.input)

    grouped = defaultdict(list)
    for item in items:
        if not args.include_unreviewed and item.get("review_status") != "reviewed":
            continue

        audio_id = item.get("audio_id") or item.get("segment_id") or "unknown_audio"
        grouped[audio_id].append(item)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for audio_id, group in grouped.items():
        group = sorted(group, key=lambda x: x.get("word_index", x.get("index", 0)))

        tokens = [x.get("corrected_text", "") for x in group]
        text = clean_join(tokens)

        first = group[0]
        audio_filepath = (
            first.get("audio_filepath")
            or first.get("audio_path")
            or args.audio_filepath
        )

        if not audio_filepath:
            raise ValueError(
                f"No audio_filepath found for audio_id={audio_id}. "
                "Pass --audio-filepath."
            )

        duration = (
            first.get("duration")
            or args.duration
        )

        if duration is None:
            starts = [
                x.get("local_start")
                for x in group
                if isinstance(x.get("local_start"), (int, float))
            ]
            ends = [
                x.get("local_end")
                for x in group
                if isinstance(x.get("local_end"), (int, float))
            ]
            if starts and ends:
                duration = max(ends) - min(starts)

        if duration is None or duration <= 0:
            raise ValueError(
                f"No valid duration found for audio_id={audio_id}. "
                "Pass --duration."
            )

        row = {
            "audio_filepath": audio_filepath,
            "duration": duration,
            "text": text,
        }

        if args.offset is not None:
            row["offset"] = args.offset

        rows.append(row)

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
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
