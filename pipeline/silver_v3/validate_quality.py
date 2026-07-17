from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_VIEWS = (
    "whole",
    "canonical_20s",
    "context_10s_stride_5s",
    "local_2p5s_contiguous",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_token(token: dict[str, Any]) -> str:
    value = str(token.get("normalized") or token.get("text") or "").strip().casefold()
    return value


def repeated_ngrams(rows: list[dict[str, Any]], n: int) -> tuple[int, list[dict[str, Any]]]:
    tokens: list[str] = []
    locations: list[tuple[int, int]] = []
    for row in rows:
        segment_position = int(row.get("segment_position", row.get("segment_index", 0)))
        for token_position, token in enumerate(row.get("tokens", [])):
            value = normalized_token(token)
            if value:
                tokens.append(value)
                locations.append((segment_position, token_position))

    occurrences: dict[tuple[str, ...], list[tuple[int, int]]] = defaultdict(list)
    for index in range(max(0, len(tokens) - n + 1)):
        gram = tuple(tokens[index:index + n])
        occurrences[gram].append(locations[index])

    repeated = [
        {
            "ngram": " ".join(gram),
            "occurrence_count": len(points),
            "extra_occurrence_count": len(points) - 1,
            "locations": [
                {"segment_position": segment, "token_position": token}
                for segment, token in points[:20]
            ],
        }
        for gram, points in occurrences.items()
        if len(points) > 1
    ]
    repeated.sort(key=lambda item: (-item["extra_occurrence_count"], item["ngram"]))
    return sum(item["extra_occurrence_count"] for item in repeated), repeated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply hard Silver v3 transcript-quality gates.")
    parser.add_argument("--segment-jsonl", type=Path, required=True)
    parser.add_argument("--normalization-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-a-tier-ratio", type=float, default=0.60)
    parser.add_argument("--max-repeated-6grams", type=int, default=25)
    parser.add_argument("--pilot-segments", type=int, default=30)
    parser.add_argument("--pilot-min-a-tier-ratio", type=float, default=0.75)
    parser.add_argument("--pilot-max-repeated-6grams", type=int, default=0)
    parser.add_argument("--require-anchor-contribution", action="store_true", default=True)
    return parser.parse_args()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tier_counts: Counter[str] = Counter()
    view_counts: Counter[str] = Counter()
    timing_counts: Counter[str] = Counter()
    total_tokens = 0

    for row in rows:
        for token in row.get("tokens", []):
            total_tokens += 1
            tier_counts[str(token.get("acceptance_tier", "unknown"))] += 1
            for view in token.get("views", []):
                view_counts[str(view)] += 1
            for observation in token.get("observations", []):
                timing_counts[str(observation.get("timing_method", "unknown"))] += 1

    a_count = tier_counts.get("A_corroborated", 0)
    return {
        "segment_count": len(rows),
        "total_tokens": total_tokens,
        "tier_distribution": dict(tier_counts),
        "a_tier_ratio": round(a_count / total_tokens, 6) if total_tokens else 0.0,
        "view_contribution_counts": dict(view_counts),
        "observation_timing_method_counts": dict(timing_counts),
    }


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.segment_jsonl)
    normalization = json.loads(args.normalization_report.read_text(encoding="utf-8"))

    full = summarize(rows)
    full_repeat_count, full_repeats = repeated_ngrams(rows, 6)
    full["repeated_6gram_extra_occurrence_count"] = full_repeat_count

    pilot_rows = rows[: min(args.pilot_segments, len(rows))]
    pilot = summarize(pilot_rows)
    pilot_repeat_count, pilot_repeats = repeated_ngrams(pilot_rows, 6)
    pilot["repeated_6gram_extra_occurrence_count"] = pilot_repeat_count

    normalization_views = normalization.get("views", {})
    view_input_checks = {}
    for view in REQUIRED_VIEWS:
        report = normalization_views.get(view, {})
        view_input_checks[view] = {
            "present": view in normalization_views,
            "normalized_row_count": int(report.get("normalized_row_count", 0)),
            "normalized_word_count": int(report.get("normalized_word_count", report.get("total_word_count", 0))),
            "source_accounting_closed": bool(report.get("source_accounting_closed", False)),
            "passed": bool(report.get("passed", False)),
        }

    required_input_views_pass = all(
        check["present"]
        and check["normalized_row_count"] > 0
        and check["normalized_word_count"] > 0
        and check["source_accounting_closed"]
        and check["passed"]
        for check in view_input_checks.values()
    )

    contribution = full["view_contribution_counts"]
    anchor_contribution_pass = (
        contribution.get("canonical_20s", 0) > 0
        and contribution.get("whole_slice", 0) > 0
    )

    gates = {
        "normalization_passed": bool(normalization.get("passed", False)),
        "required_input_views_pass": required_input_views_pass,
        "full_a_tier_ratio_pass": full["a_tier_ratio"] >= args.min_a_tier_ratio,
        "full_repeated_6grams_pass": full_repeat_count <= args.max_repeated_6grams,
        "pilot_a_tier_ratio_pass": pilot["a_tier_ratio"] >= args.pilot_min_a_tier_ratio,
        "pilot_repeated_6grams_pass": pilot_repeat_count <= args.pilot_max_repeated_6grams,
        "anchor_contribution_pass": anchor_contribution_pass if args.require_anchor_contribution else True,
        "segments_present": len(rows) > 0,
        "tokens_present": full["total_tokens"] > 0,
    }
    passed = all(gates.values())

    report = {
        "schema_version": "silver_v3_quality_validation_v1",
        "segment_jsonl": str(args.segment_jsonl),
        "normalization_report": str(args.normalization_report),
        "thresholds": {
            "min_a_tier_ratio": args.min_a_tier_ratio,
            "max_repeated_6grams": args.max_repeated_6grams,
            "pilot_segments": args.pilot_segments,
            "pilot_min_a_tier_ratio": args.pilot_min_a_tier_ratio,
            "pilot_max_repeated_6grams": args.pilot_max_repeated_6grams,
            "require_anchor_contribution": args.require_anchor_contribution,
        },
        "view_input_checks": view_input_checks,
        "full_lecture": full,
        "pilot_window": pilot,
        "top_full_repeated_6grams": full_repeats[:100],
        "top_pilot_repeated_6grams": pilot_repeats[:100],
        "gates": gates,
        "passed": passed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
