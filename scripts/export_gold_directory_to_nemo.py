from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.gold_dataset import discover_gold_files, evaluate_gold_document
from src.datasets.segment_metadata import SegmentMetadataResolver


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-export eligible Gold documents to one NeMo manifest."
    )
    parser.add_argument("--gold-dir", default=None)
    parser.add_argument("--gold-file", action="append", default=[])
    parser.add_argument("--segment-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing outputs and continue with valid segments when "
            "some Gold files fail eligibility."
        ),
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    report_path = Path(args.report)
    repo_root = Path(args.repo_root)

    existing = [path for path in [output_path, report_path] if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Output already exists. Use --force to overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    gold_files = discover_gold_files(args.gold_dir, args.gold_file)
    if not gold_files:
        raise ValueError("No Gold JSON files were selected.")

    resolver = SegmentMetadataResolver(args.segment_manifest)

    rows: list[dict] = []
    accepted: list[dict] = []
    excluded: list[dict] = []
    seen_segment_ids: set[str] = set()

    for gold_path in gold_files:
        try:
            _, evaluation = evaluate_gold_document(gold_path)
        except Exception as exc:
            excluded.append(
                {
                    "gold_file": str(gold_path),
                    "errors": [
                        {
                            "code": "gold_read_error",
                            "message": str(exc),
                        }
                    ],
                }
            )
            continue

        segment_id = evaluation.get("segment_id")

        if evaluation["eligible"] and segment_id in seen_segment_ids:
            evaluation["eligible"] = False
            evaluation["errors"].append(
                {
                    "code": "duplicate_segment_id",
                    "message": f"Duplicate Gold export for {segment_id}.",
                }
            )

        metadata = None
        if evaluation["eligible"]:
            metadata = resolver.resolve(segment_id=segment_id)

            if metadata is None:
                evaluation["eligible"] = False
                evaluation["errors"].append(
                    {
                        "code": "unresolved_segment",
                        "message": (
                            f"Segment does not resolve in the segment manifest: "
                            f"{segment_id}"
                        ),
                    }
                )

        if evaluation["eligible"]:
            audio_filepath = str(metadata["audio_filepath"])
            audio_path = Path(audio_filepath)
            check_path = (
                audio_path
                if audio_path.is_absolute()
                else repo_root / audio_path
            )

            if not check_path.exists():
                evaluation["eligible"] = False
                evaluation["errors"].append(
                    {
                        "code": "missing_audio",
                        "message": f"Audio file does not exist: {audio_filepath}",
                    }
                )

        if not evaluation["eligible"]:
            excluded.append(evaluation)
            continue

        duration = float(metadata["duration"])
        row = {
            "audio_filepath": str(metadata["audio_filepath"]),
            "duration": round(duration, 3),
            "text": evaluation["text"],
        }

        rows.append(row)
        seen_segment_ids.add(str(segment_id))
        accepted.append(
            {
                "gold_file": str(gold_path),
                "segment_id": segment_id,
                "audio_filepath": row["audio_filepath"],
                "duration": row["duration"],
                "characters": len(row["text"]),
                "nonempty_rows": evaluation["nonempty_rows"],
            }
        )

    report = {
        "gold_files_selected": len(gold_files),
        "manifest_rows_written": len(rows),
        "accepted_segments": len(accepted),
        "excluded_segments": len(excluded),
        "force": args.force,
        "segment_manifest": args.segment_manifest,
        "output": str(output_path),
        "accepted": accepted,
        "excluded": excluded,
    }

    write_json(report_path, report)

    if excluded and not args.force:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise ValueError(
            f"{len(excluded)} Gold files failed eligibility. "
            "Review the report or rerun with --force to export the valid subset."
        )

    if not rows:
        raise ValueError("No valid training rows remain after validation.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
