from __future__ import annotations

import argparse
import json
from pathlib import Path

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
EXCLUDED_PARTS = {
    "audio_views",
    "bronze_v3",
    "silver_v3",
    "silver_plus_v3",
    "silver_plus_v4",
    "fifth_view_integrated",
    "raw_parakeet",
    "normalized",
}


def score(path: Path, lecture_id: str, lecture_root: Path) -> tuple[int, int, str]:
    name = path.stem.lower()
    lecture = lecture_id.lower()
    inside_lecture_root = 0 if lecture_root in path.parents else 1
    exact = 0 if name == lecture else 1
    starts = 0 if name.startswith(lecture) else 1
    contains = 0 if lecture in name else 1
    return (inside_lecture_root * 100 + exact * 10 + starts * 3 + contains, len(str(path)), str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve one source audio file for a lecture deterministically.")
    parser.add_argument("--lecture-id", required=True)
    parser.add_argument("--lecture-root", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roots = [args.lecture_root, args.drive_root]
    candidates: list[Path] = []
    seen: set[Path] = set()

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            if any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            if args.lecture_id.lower() not in path.name.lower() and root != args.lecture_root:
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)

    if not candidates:
        raise FileNotFoundError(
            f"No source audio found for {args.lecture_id} under {args.lecture_root} or {args.drive_root}"
        )

    candidates.sort(key=lambda path: score(path, args.lecture_id, args.lecture_root.resolve()))
    selected = candidates[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(str(selected) + "\n", encoding="utf-8")

    print(json.dumps({
        "lecture_id": args.lecture_id,
        "selected_audio": str(selected),
        "candidate_count": len(candidates),
        "candidates": [str(path) for path in candidates[:20]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
