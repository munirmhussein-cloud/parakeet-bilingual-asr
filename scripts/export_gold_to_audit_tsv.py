import argparse
import csv
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
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "segment_id",
        "token_index",
        "timestamp",
        "reviewer_id",
        "field",
        "old_value",
        "new_value",
        "source"
    ]

    rows = []
    for segment in iter_gold(args.input):
        for token in segment.get("tokens", []):
            for audit in token.get("audit_history", []):
                rows.append({
                    "segment_id": segment.get("segment_id", ""),
                    "token_index": token.get("index", ""),
                    "timestamp": audit.get("timestamp", ""),
                    "reviewer_id": audit.get("reviewer_id", segment.get("reviewer_id", "")),
                    "field": audit.get("field", ""),
                    "old_value": audit.get("old_value", ""),
                    "new_value": audit.get("new_value", ""),
                    "source": audit.get("source", "")
                })

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} audit row(s): {out_path}")


if __name__ == "__main__":
    main()
