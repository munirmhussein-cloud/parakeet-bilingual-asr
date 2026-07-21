from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.reconcile_silver_v3_lattice import (
    _refresh_resolved_token_evidence,
    export_docx,
    join_tokens,
    sha256_file,
    write_jsonl,
)

REPAIR_SCHEMA = "silver_v3_validator_semantic_duplicate_repair_v1"
MAX_PAIR_DELTA = 2.5
MAX_DELTA_SPREAD = 0.55
MIN_VIEW_JACCARD = 0.50
MIN_COMMON_VIEWS = 2
MIN_LEXICAL_PAIRS = 4


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"Non-object JSONL row at {path}:{line_number}"
                )
            rows.append(value)
    if not rows:
        raise RuntimeError(f"JSONL contains no rows: {path}")
    return rows


def validator_normalized_token(token: dict[str, Any]) -> str:
    return str(
        token.get("normalized") or token.get("text") or ""
    ).strip().casefold()


def immediate_duplicate_cases(
    tokens: list[dict[str, Any]],
    *,
    n: int = 6,
) -> list[dict[str, Any]]:
    visible = [
        (position, validator_normalized_token(token))
        for position, token in enumerate(tokens)
        if validator_normalized_token(token)
    ]
    cases: list[dict[str, Any]] = []
    for start in range(max(0, len(visible) - (2 * n) + 1)):
        first = visible[start:start + n]
        second = visible[start + n:start + (2 * n)]
        if [value for _, value in first] != [
            value for _, value in second
        ]:
            continue
        cases.append(
            {
                "ngram": " ".join(value for _, value in first),
                "first_positions": [position for position, _ in first],
                "second_positions": [position for position, _ in second],
            }
        )
    return cases


def safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def token_views(token: dict[str, Any]) -> set[str]:
    views = token.get("views", [])
    if not isinstance(views, list):
        return set()
    return {str(view) for view in views if str(view)}


def evaluate_case(
    tokens: list[dict[str, Any]],
    case: dict[str, Any],
    *,
    maximum_pair_center_delta: float = MAX_PAIR_DELTA,
    maximum_pair_delta_spread: float = MAX_DELTA_SPREAD,
    minimum_pair_view_jaccard: float = MIN_VIEW_JACCARD,
    minimum_common_view_count: int = MIN_COMMON_VIEWS,
    minimum_lexical_pair_count: int = MIN_LEXICAL_PAIRS,
) -> dict[str, Any]:
    """
    Accept an exact immediate duplicate when corresponding lexical tokens
    form either a same-time overlap or a rigid temporal translation.

    View sets may gain or lose a witness at window boundaries. Evidence is
    therefore based on substantial shared-view support, not exact equality.
    """
    first_positions = list(case["first_positions"])
    second_positions = list(case["second_positions"])
    raw_gap = second_positions[0] - first_positions[-1] - 1
    reasons: list[str] = []
    if raw_gap > 2:
        reasons.append(f"raw_gap_too_large:{raw_gap}")

    deltas: list[float] = []
    jaccards: list[float] = []
    common_counts: list[int] = []
    pair_details: list[dict[str, Any]] = []
    lexical_pairs = 0

    for first_position, second_position in zip(
        first_positions,
        second_positions,
    ):
        first = tokens[first_position]
        second = tokens[second_position]
        lexical = bool(
            str(first.get("normalized") or "").strip()
            or str(second.get("normalized") or "").strip()
        )
        first_center = safe_float(first.get("center"))
        second_center = safe_float(second.get("center"))
        signed_delta = (
            second_center - first_center
            if first_center is not None and second_center is not None
            else None
        )
        delta = abs(signed_delta) if signed_delta is not None else None

        first_views = token_views(first)
        second_views = token_views(second)
        common = first_views & second_views
        union = first_views | second_views
        jaccard = len(common) / len(union) if union else 0.0

        if lexical:
            lexical_pairs += 1
            if signed_delta is None:
                reasons.append(
                    f"missing_center:{first_position}:{second_position}"
                )
            else:
                deltas.append(delta)
                if signed_delta < -0.05:
                    reasons.append(
                        "pair_center_order_reversed:"
                        f"{first_position}:{second_position}:"
                        f"{signed_delta:.6f}"
                    )
                if delta > maximum_pair_center_delta:
                    reasons.append(
                        "pair_center_delta_exceeds_limit:"
                        f"{first_position}:{second_position}:{delta:.6f}"
                    )

            jaccards.append(jaccard)
            common_counts.append(len(common))
            same_nonempty_set = bool(first_views) and (
                first_views == second_views
            )
            enough_shared_support = (
                len(common) >= minimum_common_view_count
                and jaccard >= minimum_pair_view_jaccard
            )
            if not (same_nonempty_set or enough_shared_support):
                reasons.append(
                    "pair_view_overlap_below_limit:"
                    f"{first_position}:{second_position}:"
                    f"common={len(common)}:jaccard={jaccard:.6f}"
                )

        pair_details.append(
            {
                "first_position": first_position,
                "second_position": second_position,
                "token": validator_normalized_token(first),
                "evidence_pair": lexical,
                "first_center": first_center,
                "second_center": second_center,
                "signed_center_delta": signed_delta,
                "center_delta": delta,
                "first_views": sorted(first_views),
                "second_views": sorted(second_views),
                "common_views": sorted(common),
                "view_jaccard": round(jaccard, 6),
            }
        )

    if lexical_pairs < minimum_lexical_pair_count:
        reasons.append(
            f"insufficient_lexical_pairs:"
            f"{lexical_pairs}<{minimum_lexical_pair_count}"
        )

    minimum_delta = min(deltas) if deltas else None
    maximum_delta = max(deltas) if deltas else None
    mean_delta = sum(deltas) / len(deltas) if deltas else None
    spread = (
        maximum_delta - minimum_delta
        if maximum_delta is not None and minimum_delta is not None
        else None
    )
    if spread is not None and spread > maximum_pair_delta_spread:
        reasons.append(
            f"pair_center_delta_spread_exceeds_limit:{spread:.6f}"
        )

    return {
        **case,
        "evidence_model":
            "exact_adjacent_same_time_or_rigid_translation_v2",
        "raw_gap": raw_gap,
        "lexical_pair_count": lexical_pairs,
        "minimum_pair_center_delta": minimum_delta,
        "maximum_pair_center_delta": maximum_delta,
        "mean_pair_center_delta": mean_delta,
        "pair_center_delta_spread": spread,
        "minimum_pair_view_jaccard": min(jaccards) if jaccards else None,
        "minimum_common_view_count":
            min(common_counts) if common_counts else None,
        "pair_details": pair_details,
        "eligible": not reasons,
        "ineligible_reasons": sorted(set(reasons)),
    }


def collect_evidence_ids(rows: list[dict[str, Any]]) -> set[str]:
    output: set[str] = set()
    for row in rows:
        for token in row.get("tokens", []):
            for observation in token.get("observations", []):
                if not isinstance(observation, dict):
                    continue
                observation_id = str(
                    observation.get("observation_id", "")
                )
                if observation_id:
                    output.add(observation_id)
            for alternate in token.get("alternates", []):
                if not isinstance(alternate, dict):
                    continue
                output.update(
                    str(item)
                    for item in alternate.get("observation_ids", [])
                    if str(item)
                )
    return output


def collapse_case(
    tokens: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if evaluation.get("eligible") is not True:
        raise ValueError("Cannot collapse an ineligible duplicate case")

    output = list(tokens)
    first_positions = list(evaluation["first_positions"])
    second_positions = list(evaluation["second_positions"])
    removed_surfaces: list[str] = []

    for first_position, second_position in zip(
        first_positions,
        second_positions,
    ):
        retained = output[first_position]
        duplicate = output[second_position]
        retained.setdefault("observations", []).extend(
            duplicate.get("observations", [])
        )
        retained.setdefault("alternates", []).extend(
            duplicate.get("alternates", [])
        )
        _refresh_resolved_token_evidence(retained)
        removed_surfaces.append(str(duplicate.get("text", "")))

    for position in sorted(second_positions, reverse=True):
        del output[position]
    for position, token in enumerate(output):
        token["slot_id"] = position

    return output, {
        "validator_ngram": evaluation["ngram"],
        "normalized_ngram": evaluation["ngram"].split(" "),
        "first_raw_start": first_positions[0],
        "second_raw_start": second_positions[0],
        "removed_raw_positions": second_positions,
        "removed_token_count": len(second_positions),
        "removed_surfaces": removed_surfaces,
        "evidence_model": evaluation["evidence_model"],
        "minimum_pair_center_delta":
            evaluation["minimum_pair_center_delta"],
        "maximum_pair_center_delta":
            evaluation["maximum_pair_center_delta"],
        "mean_pair_center_delta":
            evaluation["mean_pair_center_delta"],
        "pair_center_delta_spread":
            evaluation["pair_center_delta_spread"],
        "minimum_pair_view_jaccard":
            evaluation["minimum_pair_view_jaccard"],
        "minimum_common_view_count":
            evaluation["minimum_common_view_count"],
        "pair_details": evaluation["pair_details"],
        "reason":
            "validator_semantic_adjacent_rigid_translation_artifact",
    }


def refresh_row(row: dict[str, Any]) -> None:
    tokens = row.get("tokens", [])
    if not isinstance(tokens, list):
        raise RuntimeError("Segment tokens is not a list")
    for position, token in enumerate(tokens):
        token["slot_id"] = position

    silver_text = join_tokens(
        [str(token.get("text", "")) for token in tokens]
    )
    row["silver_text"] = silver_text
    row["has_silver_text"] = bool(silver_text)
    row["token_count"] = len(tokens)
    row["tier_counts"] = dict(
        Counter(
            str(token.get("acceptance_tier", "unknown"))
            for token in tokens
        )
    )
    row["view_contribution_counts"] = dict(
        Counter(
            str(view)
            for token in tokens
            for view in token.get("views", [])
        )
    )
    row["immediate_duplicate_6gram_count"] = len(
        immediate_duplicate_cases(tokens)
    )
    collapses = row.get("duplicate_overlap_collapses", [])
    if not isinstance(collapses, list):
        collapses = []
    row["duplicate_overlap_collapses"] = collapses
    row["duplicate_overlap_collapse_count"] = len(collapses)


def scan_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        tokens = row.get("tokens", [])
        if not isinstance(tokens, list):
            raise RuntimeError(
                f"Segment row {row_index} tokens is not a list"
            )
        for case in immediate_duplicate_cases(tokens):
            cases.append(
                {
                    "row_index": row_index,
                    "segment_position": int(
                        row.get("segment_position", row_index)
                    ),
                    "segment_id": row.get("segment_id"),
                    **evaluate_case(tokens, case),
                }
            )
    return cases


def repair_rows(
    rows: list[dict[str, Any]],
    *,
    maximum_remaining: int = 1,
) -> dict[str, Any]:
    if maximum_remaining < 0:
        raise ValueError("maximum_remaining must be non-negative")

    before_ids = collect_evidence_ids(rows)
    before_cases = scan_rows(rows)
    collapses: list[dict[str, Any]] = []

    while len(scan_rows(rows)) > maximum_remaining:
        eligible = [
            case
            for case in scan_rows(rows)
            if case.get("eligible") is True
        ]
        if not eligible:
            break
        chosen = min(
            eligible,
            key=lambda case: (
                float(case["pair_center_delta_spread"]),
                -float(case["minimum_pair_view_jaccard"]),
                float(case["maximum_pair_center_delta"]),
                int(case["segment_position"]),
                int(case["first_positions"][0]),
            ),
        )
        row = rows[int(chosen["row_index"])]
        row["tokens"], collapse = collapse_case(
            row["tokens"],
            chosen,
        )
        row.setdefault("duplicate_overlap_collapses", []).append(
            collapse
        )
        refresh_row(row)
        collapses.append(
            {
                "segment_position": chosen["segment_position"],
                "segment_id": chosen["segment_id"],
                **collapse,
            }
        )

    for row in rows:
        refresh_row(row)

    after_cases = scan_rows(rows)
    after_ids = collect_evidence_ids(rows)
    if before_ids != after_ids:
        raise RuntimeError(
            "Duplicate repair violated evidence accounting: "
            f"missing={sorted(before_ids - after_ids)[:20]} "
            f"added={sorted(after_ids - before_ids)[:20]}"
        )

    return {
        "before_count": len(before_cases),
        "after_count": len(after_cases),
        "maximum_remaining": maximum_remaining,
        "collapse_count": len(collapses),
        "collapses": collapses,
        "remaining_cases": after_cases,
        "before_cases": before_cases,
        "evidence_id_count": len(before_ids),
        "evidence_accounting_preserved": True,
        "within_quality_limit":
            len(after_cases) <= maximum_remaining,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-jsonl", type=Path, required=True)
    parser.add_argument(
        "--token-provenance-jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument("--export-json", type=Path, required=True)
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument(
        "--reconciliation-report",
        type=Path,
        required=True,
    )
    parser.add_argument("--title", required=True)
    parser.add_argument("--maximum-remaining", type=int, default=1)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = read_jsonl(args.segment_jsonl)
    rows = copy.deepcopy(source_rows)
    reconciliation = read_json(args.reconciliation_report)
    positions = [row.get("segment_position") for row in rows]
    segment_ids = [row.get("segment_id") for row in rows]

    repair = repair_rows(
        rows,
        maximum_remaining=args.maximum_remaining,
    )
    if [row.get("segment_position") for row in rows] != positions:
        raise RuntimeError("Duplicate repair changed segment positions")
    if [row.get("segment_id") for row in rows] != segment_ids:
        raise RuntimeError("Duplicate repair changed segment IDs")

    passed = (
        repair["within_quality_limit"] is True
        and repair["evidence_accounting_preserved"] is True
    )

    if passed and repair["collapse_count"] > 0:
        write_jsonl(args.segment_jsonl, rows)
        provenance_rows = [
            {
                "schema_version":
                    f"{row.get('schema_version', '')}_token_provenance",
                "segment_position": row["segment_position"],
                "segment_id": row["segment_id"],
                **token,
            }
            for row in rows
            for token in row.get("tokens", [])
        ]
        write_jsonl(args.token_provenance_jsonl, provenance_rows)

        args.export_json.write_text(
            json.dumps(
                {
                    "schema_version":
                        f"{rows[0].get('schema_version', '')}_export",
                    "lecture_id": rows[0].get("lecture_id"),
                    "segment_count": len(rows),
                    "segments": rows,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        export_docx(rows, args.docx, title_text=args.title)

        validation = reconciliation.setdefault("validation", {})
        validation["immediate_duplicate_6gram_count"] = (
            repair["after_count"]
        )
        validation["duplicate_overlap_collapse_count"] = sum(
            int(row.get("duplicate_overlap_collapse_count", 0))
            for row in rows
        )
        validation["zero_drop_invariant"] = True
        validation["unaccounted_observation_count"] = 0
        validation["passed"] = (
            validation.get("segment_positions_ordered") is True
            and validation.get("segment_ids_unique") is True
            and int(validation.get("chronology_error_count", -1)) == 0
            and repair["after_count"] == 0
        )
        reconciliation["total_tokens"] = sum(
            int(row.get("token_count", 0)) for row in rows
        )
        reconciliation["tier_distribution"] = dict(
            Counter(
                str(token.get("acceptance_tier", "unknown"))
                for row in rows
                for token in row.get("tokens", [])
            )
        )
        reconciliation["single_witness_token_count"] = sum(
            token.get("single_witness") is True
            for row in rows
            for token in row.get("tokens", [])
        )
        reconciliation["outputs"] = {
            "segment_jsonl": str(args.segment_jsonl),
            "token_provenance_jsonl":
                str(args.token_provenance_jsonl),
            "json": str(args.export_json),
            "docx": str(args.docx),
        }
        reconciliation["sha256"] = {
            "segment_jsonl": sha256_file(args.segment_jsonl),
            "token_provenance_jsonl":
                sha256_file(args.token_provenance_jsonl),
            "json": sha256_file(args.export_json),
            "docx": sha256_file(args.docx),
        }
        reconciliation["validator_semantic_duplicate_repair"] = {
            "schema_version": REPAIR_SCHEMA,
            **repair,
        }
        args.reconciliation_report.write_text(
            json.dumps(
                reconciliation,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    report = {
        "schema_version": REPAIR_SCHEMA,
        "segment_jsonl": str(args.segment_jsonl),
        "token_provenance_jsonl":
            str(args.token_provenance_jsonl),
        "export_json": str(args.export_json),
        "docx": str(args.docx),
        "reconciliation_report":
            str(args.reconciliation_report),
        "transactional_write_performed":
            passed and repair["collapse_count"] > 0,
        **repair,
        "passed": passed,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
