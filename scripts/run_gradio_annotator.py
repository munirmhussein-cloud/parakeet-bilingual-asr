import argparse
from pathlib import Path

from apps.gradio_annotator import build_app


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio bilingual annotation app.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--progress", default="data/annotations/gradio_progress_v1.json")
    parser.add_argument("--events", default="data/annotations/gradio_events_v1.jsonl")
    parser.add_argument("--gold-json", default="data/annotations/gold_annotations_v1.json")
    parser.add_argument("--gold-jsonl", default="data/annotations/gold_annotations_v1.jsonl")
    parser.add_argument("--share", action="store_true", default=True)
    args = parser.parse_args()

    for path in [
        args.progress,
        args.events,
        args.gold_json,
        args.gold_jsonl,
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    app = build_app(
        input_path=args.input,
        progress_path=args.progress,
        events_path=args.events,
        gold_json_path=args.gold_json,
        gold_jsonl_path=args.gold_jsonl,
    )

    app.launch(share=args.share)


if __name__ == "__main__":
    main()
