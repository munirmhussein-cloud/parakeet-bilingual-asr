import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="schemas/gold_annotation_v1.schema.json")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    schema = load_json(args.schema)
    validator = Draft202012Validator(schema)

    path = Path(args.input)
    records = []

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        records = [load_json(path)]

    errors = []
    for i, record in enumerate(records):
        for error in validator.iter_errors(record):
            errors.append((i, list(error.path), error.message))

    if errors:
        print(f"Validation errors: {len(errors)}")
        for i, path, message in errors:
            print(f"[record {i}] {'.'.join(map(str, path))}: {message}")
        raise SystemExit(1)

    print(f"Validation passed: {len(records)} record(s)")


if __name__ == "__main__":
    main()
