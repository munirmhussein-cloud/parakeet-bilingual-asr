from __future__ import annotations

import argparse
import json
import re
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
LECTURE_ID_RE = re.compile(r"^lecture_(\d+)$", re.IGNORECASE)
SEERAH_NUMBER_RE = re.compile(r"^seerah\D*0*(\d{1,3})(?!\d)", re.IGNORECASE)


def lecture_number(lecture_id: str) -> int:
    match = LECTURE_ID_RE.fullmatch(lecture_id.strip())
    if not match:
        raise ValueError(f"Unsupported lecture ID format: {lecture_id!r}; expected lecture_XXX")
    return int(match.group(1))


def audio_number(path: Path) -> int | None:
    match = SEERAH_NUMBER_RE.match(path.stem)
    return int(match.group(1)) if match else None


def matches_lecture(path: Path, lecture_id: str) -> bool:
    return audio_number(path) == lecture_number(lecture_id)


def score(path: Path, lecture_id: str, lecture_root: Path) -> tuple[int, int, int, str]:
    name = path.stem.lower()
    lecture = lecture_id.lower()
    inside_lecture_root = 0 if lecture_root in path.parents else 1
    literal = 0 if lecture in name else 1
    compact_seerah = 0 if re.match(r"^seerah\d", name) else 1
    return (inside_lecture_root, literal, compact_seerah, len(str(path)), str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve one source audio file for a lecture deterministically.")
    parser.add_argument("--lecture-id", required=True)
    parser.add_argument("--lecture-root", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requested_number = lecture_number(args.lecture_id)
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
            if not matches_lecture(path, args.lecture_id):
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)

    if not candidates:
        raise FileNotFoundError(
            f"No Seerah source audio numbered {requested_number} found for {args.lecture_id} "
            f"under {args.lecture_root} or {args.drive_root}"
        )

    candidates.sort(key=lambda path: score(path, args.lecture_id, args.lecture_root.resolve()))
    selected = candidates[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(str(selected) + "\n", encoding="utf-8")

    print(json.dumps({
        "lecture_id": args.lecture_id,
        "lecture_number": requested_number,
        "selected_audio": str(selected),
        "selected_audio_number": audio_number(selected),
        "candidate_count": len(candidates),
        "candidates": [str(path) for path in candidates[:20]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
