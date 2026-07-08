import argparse
import json
from pathlib import Path


def iter_gold(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            yield from data
        else:
            yield data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-field", default="gold_text")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in iter_gold(args.input):
            manifest = {
                "audio_filepath": row["audio_filepath"],
                "duration": row["duration"],
                "text": row.get(args.text_field, row["gold_text"])
            }
            if "offset" in row:
                manifest["offset"] = row["offset"]

            out.write(json.dumps(manifest, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} NeMo manifest row(s): {out_path}")


if __name__ == "__main__":
    main()
