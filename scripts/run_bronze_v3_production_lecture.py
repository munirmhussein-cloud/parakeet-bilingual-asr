from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def phrase_span(phrase: dict[str, Any]) -> tuple[float, float]:
    start_ms = number(phrase.get("offsetMilliseconds"), -1.0)
    duration_ms = number(phrase.get("durationMilliseconds"), -1.0)
    if start_ms >= 0:
        start = start_ms / 1000.0
        end = start + max(0.0, duration_ms) / 1000.0
        return start, end
    start = number(phrase.get("offset"), number(phrase.get("start"), 0.0))
    duration = number(phrase.get("duration"), 0.0)
    end = number(phrase.get("end"), start + duration)
    return start, max(start, end)


def phrase_text(phrase: dict[str, Any]) -> str:
    return str(
        phrase.get("text")
        or phrase.get("displayText")
        or phrase.get("lexicalText")
        or ""
    ).strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def build_parent_rows(raw: dict[str, Any], lecture_id: str) -> list[dict[str, Any]]:
    response = raw.get("response", {}) if isinstance(raw.get("response"), dict) else {}
    phrases = response.get("phrases", []) if isinstance(response.get("phrases"), list) else []
    duration_ms = number(response.get("durationMilliseconds"), 0.0)
    duration = duration_ms / 1000.0 if duration_ms > 0 else 0.0
    if duration <= 0:
        duration = max((phrase_span(item)[1] for item in phrases if isinstance(item, dict)), default=0.0)
    parent_count = max(1, int((duration + 19.999999) // 20.0))
    parents: list[dict[str, Any]] = []
    for position in range(parent_count):
        start = position * 20.0
        end = min(duration, start + 20.0) if duration > 0 else start + 20.0
        children = []
        for phrase in phrases:
            if not isinstance(phrase, dict):
                continue
            phrase_start, phrase_end = phrase_span(phrase)
            if phrase_end <= start or phrase_start >= end:
                continue
            text = phrase_text(phrase)
            if not text:
                continue
            children.append({
                "transcript": text,
                "global_start": round(max(start, phrase_start), 6),
                "global_end": round(min(end, phrase_end), 6),
                "locale": phrase.get("locale"),
            })
        parents.append({
            "schema_version": "azure_bronze_v3_parent_segment_v1",
            "lecture_id": lecture_id,
            "segment_id": f"{lecture_id}__canonical_20s__{position:05d}",
            "segment_position": position,
            "segment_index": position,
            "segment_start": round(start, 6),
            "segment_end": round(end, 6),
            "duration": round(max(0.0, end - start), 6),
            "children": children,
            "text": " ".join(child["transcript"] for child in children).strip(),
        })
    return parents


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Azure whole-lecture Bronze v3 and emit canonical parent JSONL.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--lecture-id", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = output_root / f"{args.lecture_id}_bronze_v3_azure_whole.json"
    parent_path = output_root / f"{args.lecture_id}_bronze_v3_azure_parent.jsonl"

    subprocess.run([
        sys.executable, "-m", "pipeline.bronze_v3.run",
        "--lecture-id", args.lecture_id,
        "--audio", str(args.audio),
        "--output", str(raw_path),
    ], cwd=args.repo_root, check=True)

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = build_parent_rows(raw, args.lecture_id)
    write_jsonl(parent_path, rows)
    print(json.dumps({
        "completed": True,
        "lecture_id": args.lecture_id,
        "raw_output": str(raw_path),
        "parent_jsonl": str(parent_path),
        "parent_count": len(rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
