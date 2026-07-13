"""
Convenience launcher for the Sprint 2 Gradio reviewer.
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.gradio_annotator import build_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--reviewer-id", default="reviewer")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = build_app(
        input_path=args.input,
        reviewer_id=args.reviewer_id,
    )

    app.launch(
        share=args.share,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
