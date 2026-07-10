from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.segment_metadata import SegmentMetadataResolver


def assign_split(
    group_id: str,
    seed: str,
    train_ratio: float,
    dev_ratio: float,
) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()
    fraction = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)

    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + dev_ratio:
        return "dev"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically split a NeMo dataset by source recording."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--segment-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", default="42")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ratio_total = args.train_ratio + args.dev_ratio + args.test_ratio
    if abs(ratio_total - 1.0) > 1e-9:
        raise ValueError("Train/dev/test ratios must sum to 1.0.")

    output_dir = Path(args.output_dir)
    outputs = {
        name: output_dir / f"{name}.jsonl"
        for name in ["train", "dev", "test"]
    }
    report_path = output_dir / "split_report.json"

    existing = [
        path
        for path in [*outputs.values(), report_path]
        if path.exists()
    ]
    if existing and not args.force:
        raise FileExistsError(
            "Split output exists. Use --force to overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    resolver = SegmentMetadataResolver(args.segment_manifest)
    rows = []

    for line_number, line in enumerate(
        Path(args.input).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        row = json.loads(line)
        metadata = resolver.resolve(
            audio_filepath=str(row.get("audio_filepath", ""))
        )

        if metadata is None:
            raise KeyError(
                f"Input row {line_number} does not resolve in segment manifest."
            )

        source_audio_id = metadata.get("source_audio_id")
        if not source_audio_id:
            raise ValueError(
                f"Segment {metadata.get('segment_id')} is missing source_audio_id."
            )

        rows.append((row, metadata, str(source_audio_id)))

    split_rows: dict[str, list[dict]] = defaultdict(list)
    split_groups: dict[str, set[str]] = defaultdict(set)
    split_seconds: dict[str, float] = defaultdict(float)

    group_assignments = {}

    for row, metadata, group_id in rows:
        split = group_assignments.setdefault(
            group_id,
            assign_split(
                group_id,
                args.seed,
                args.train_ratio,
                args.dev_ratio,
            ),
        )

        split_rows[split].append(row)
        split_groups[split].add(group_id)
        split_seconds[split] += float(row["duration"])

    output_dir.mkdir(parents=True, exist_ok=True)

    for split, path in outputs.items():
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows.get(split, []):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "input": args.input,
        "segment_manifest": args.segment_manifest,
        "strategy": "sha256_source_audio_id",
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "dev": args.dev_ratio,
            "test": args.test_ratio,
        },
        "splits": {
            split: {
                "rows": len(split_rows.get(split, [])),
                "source_recordings": len(split_groups.get(split, set())),
                "duration_seconds": round(split_seconds.get(split, 0.0), 3),
                "hours": round(split_seconds.get(split, 0.0) / 3600.0, 4),
            }
            for split in ["train", "dev", "test"]
        },
        "group_assignments": dict(sorted(group_assignments.items())),
    }

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
