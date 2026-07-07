"""
Validate and normalize a Gradio reconciliation input file.
"""

from __future__ import annotations

import argparse
import json

from src.annotation.gradio_data_model import (
    validate_reconciliation_input,
    normalize_reconciliation_input,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    _, summary = validate_reconciliation_input(args.input)
    normalized = normalize_reconciliation_input(args.input)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(json.dumps(normalized["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
