from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from integrate_silver_plus_v3_fifth_view import (
    AZURE_VIEW,
    TOKEN_RE,
    is_arabic,
    skeleton,
    token_text,
    write_jsonl,
)
from run_silver_plus_v3_fifth_view_export_v2 import integrate as integrate_v3
from run_silver_plus_v4_fifth_view_export import (
    BUILT_ON_SPINE,
    HARAKAT,
    HONORIFIC_EQUIVALENCE,
    LAYER_VERSION,
    apply_fix_b,
    arabic_letters,
    git_head,
    provenance_rows,
    read_jsonl,
    serialize,
    sha256_file,
)

PRODUCTION_VALIDATION_SCHEMA = "silver_plus_v4_production_validation_v1"
PILOT_SEGMENTS = 30
MIN_PILOT_AR_RECALL = 0.33
MIN_PILOT_EN_RECALL = 0.80
MIN_TASHKEEL_DENSITY = 0.20
MIN_CORROBORATION_SHARE = 0.60
MAX_IMMEDIATE_DUPLICATE_6GRAMS = 5
REQUIRED_VIEWS = (
    "canonical_20s",
    "whole_slice",
    "context_10s_stride_5s",
    "local_2p5s_contiguous",
    AZURE_VIEW,
)
ANCHOR_VIEWS = {"canonical_20s", "whole_slice"}


def canonical_view(value: Any) -> str:
    view = str(value or "")
    return "local_2p5s_contiguous" if view == "local_2p5s" else view


def complete_views(token: dict[str, Any]) -> list[str]:
    values = list(token.get("views", []))
    for observation in token.get("observations", []):
        if not isinstance(observation, dict):
            continue
        for key in ("view", "view_id", "source_view", "source"):
            value = observation.get(key)
            if isinstance(value, str) and value:
                values.append(value)
    return list(
        dict.fromkeys(
            canonical_view(value)
            for value in values
            if canonical_view(value)
        )
    )


def immediate_duplicate_6gram_count(
    tokens: list[dict[str, Any]],
    *,
    n: int = 6,
) -> int:
    values = [
        skeleton(token_text(token))
        for token in tokens
        if token_text(token) and skeleton(token_text(token))
    ]
    return sum(
        values[index : index + n]
        == values[index + n : index + (2 * n)]
        for index in range(max(0, len(values) - (2 * n) + 1))
    )


def finalize_segment_row(row: dict[str, Any]) -> dict[str, Any]:
    fixed = apply_fix_b(row)
    text = str(fixed.get("silver_plus_v4_text") or "")
    fixed["has_silver_plus_v4_text"] = bool(text.strip())
    tokens = fixed.get("tokens", [])
    if not isinstance(tokens, list):
        raise RuntimeError("Silver+ v4 segment tokens must be a list")
    fixed["immediate_duplicate_6gram_count"] = (
        immediate_duplicate_6gram_count(tokens)
    )
    return fixed


def lexical_tokens(text: str, *, arabic: bool) -> list[str]:
    output: list[str] = []
    for piece in TOKEN_RE.findall(str(text or "")):
        key = skeleton(piece)
        if not key:
            continue
        if arabic:
            if is_arabic(piece):
                output.append(key)
        elif any(
            character.isascii() and character.isalnum()
            for character in piece
        ):
            output.append(key)
    return output


def maximum_count_union(
    references: list[str],
    *,
    arabic: bool,
) -> Counter[str]:
    union: Counter[str] = Counter()
    for reference in references:
        observed = Counter(lexical_tokens(reference, arabic=arabic))
        for token, count in observed.items():
            union[token] = max(union[token], count)
    return union


def count_aware_recall(
    hypothesis: str,
    references: list[str],
    *,
    arabic: bool,
) -> tuple[int, int]:
    expected = maximum_count_union(references, arabic=arabic)
    observed = Counter(lexical_tokens(hypothesis, arabic=arabic))
    hits = sum(
        min(count, observed[token])
        for token, count in expected.items()
    )
    return hits, sum(expected.values())


def azure_parent_text(row: dict[str, Any]) -> str:
    children = row.get("children", [])
    if isinstance(children, list):
        values = [
            str(
                child.get("transcript") or child.get("text") or ""
            ).strip()
            for child in children
            if isinstance(child, dict)
        ]
        joined = " ".join(value for value in values if value).strip()
        if joined:
            return joined
    return str(
        row.get("azure_forced_ar_text") or row.get("text") or ""
    ).strip()


def run_production_gates(
    rows: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    azure_rows: list[dict[str, Any]],
    deterministic: bool,
) -> dict[str, Any]:
    if len(rows) != len(azure_rows):
        raise ValueError(
            f"Validation row-count mismatch: silver_plus_v4={len(rows)} "
            f"azure_parent={len(azure_rows)}"
        )

    losses: list[str] = []
    net_arabic_letters_lost = 0
    for row in rows:
        before = arabic_letters(row.get("silver_v3_text", ""))
        after = arabic_letters(row.get("silver_plus_v4_text", ""))
        if after < before:
            losses.append(
                str(
                    row.get("embedded_seg_id")
                    or row.get("segment_id")
                )
            )
            net_arabic_letters_lost += before - after

    full_text = " ".join(
        str(row.get("silver_plus_v4_text") or "") for row in rows
    )
    arabic_count = arabic_letters(full_text)
    tashkeel_density = (
        len(HARAKAT.findall(full_text)) / arabic_count
        if arabic_count
        else 0.0
    )

    ar_hits = ar_total = en_hits = en_total = 0
    pilot_count = min(PILOT_SEGMENTS, len(rows))
    for row, azure in zip(
        rows[:pilot_count],
        azure_rows[:pilot_count],
    ):
        references = [
            str(row.get("silver_v3_text") or ""),
            azure_parent_text(azure),
        ]
        hit, total = count_aware_recall(
            str(row.get("silver_plus_v4_text") or ""),
            references,
            arabic=True,
        )
        ar_hits += hit
        ar_total += total
        hit, total = count_aware_recall(
            str(row.get("silver_plus_v4_text") or ""),
            references,
            arabic=False,
        )
        en_hits += hit
        en_total += total
    ar_recall = ar_hits / ar_total if ar_total else 1.0
    en_recall = en_hits / en_total if en_total else 1.0

    total_tokens = len(provenance)
    corroborated = sum(
        row.get("tier") == "A_corroborated" for row in provenance
    )
    corroboration_share = (
        corroborated / total_tokens if total_tokens else 0.0
    )

    duplicate_count = sum(
        int(row.get("immediate_duplicate_6gram_count", 0) or 0)
        for row in rows
    )

    observed_global: set[str] = set()
    segments_missing_anchor_views: list[str] = []
    for row in rows:
        observed = {
            view
            for token in row.get("tokens", [])
            if isinstance(token, dict)
            for view in complete_views(token)
        }
        observed_global.update(observed)
        if (
            str(row.get("silver_plus_v4_text") or "").strip()
            and not (ANCHOR_VIEWS & observed)
        ):
            segments_missing_anchor_views.append(
                str(
                    row.get("segment_id")
                    or row.get("embedded_seg_id")
                )
            )
    missing_global_views = sorted(
        set(REQUIRED_VIEWS) - observed_global
    )

    gates = {
        "all_views_present": {
            "pass": not missing_global_views
            and not segments_missing_anchor_views,
            "views": list(REQUIRED_VIEWS),
            "missing_global_views": missing_global_views,
            "anchor_views": sorted(ANCHOR_VIEWS),
            "segments_missing_anchor_views": (
                segments_missing_anchor_views
            ),
        },
        "corroboration_floor": {
            "pass": (
                corroboration_share >= MIN_CORROBORATION_SHARE
            ),
            "a_tier_share": corroboration_share,
            "corroborated_tokens": corroborated,
            "total_tokens": total_tokens,
            "threshold": MIN_CORROBORATION_SHARE,
        },
        "determinism": {
            "pass": deterministic,
            "rerun_hash_matches": deterministic,
        },
        "duplicate_budget": {
            "pass": (
                duplicate_count <= MAX_IMMEDIATE_DUPLICATE_6GRAMS
            ),
            "summed_immediate_duplicate_6grams": duplicate_count,
            "threshold": MAX_IMMEDIATE_DUPLICATE_6GRAMS,
            "metric": (
                "summed_segment_immediate_duplicate_6gram_count"
            ),
        },
        "no_regression_vs_silver_v3": {
            "pass": not losses,
            "segments_losing_ar": len(losses),
            "segment_ids_losing_ar": losses,
            "net_ar_letters_lost": net_arabic_letters_lost,
        },
        "pilot_window_union_reproduction": {
            "pass": ar_recall >= MIN_PILOT_AR_RECALL
            and en_recall >= MIN_PILOT_EN_RECALL,
            "reference_model": (
                "max-count lexical union of silver_v3_text and "
                "Azure parent text"
            ),
            "pilot_segments": pilot_count,
            "ar_recall": ar_recall,
            "en_recall": en_recall,
            "ar_hit": ar_hits,
            "ar_total": ar_total,
            "en_hit": en_hits,
            "en_total": en_total,
            "ar_threshold": MIN_PILOT_AR_RECALL,
            "en_threshold": MIN_PILOT_EN_RECALL,
        },
        "vocalization_floor": {
            "pass": tashkeel_density >= MIN_TASHKEEL_DENSITY,
            "tashkeel_density": tashkeel_density,
            "threshold": MIN_TASHKEEL_DENSITY,
        },
    }
    all_gates_pass = all(
        gate.get("pass") is True for gate in gates.values()
    )
    return {
        "schema_version": PRODUCTION_VALIDATION_SCHEMA,
        "validation_performed": True,
        "all_gates_pass": all_gates_pass,
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Silver+ v4 production integration with internal "
            "production gate evaluation and downstream-ingest "
            "contract stamps."
        )
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--silver-v3", type=Path, required=True)
    parser.add_argument("--azure-parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--max-align-seconds",
        type=float,
        default=0.85,
    )
    args = parser.parse_args()

    if "silver_plus_v4" not in args.output_prefix:
        raise ValueError(
            "output-prefix must use lecture_XXX_silver_plus_v4 naming"
        )

    silver_rows = read_jsonl(args.silver_v3)
    azure_rows = read_jsonl(args.azure_parent)
    if len(silver_rows) != len(azure_rows):
        raise ValueError(
            f"Row count mismatch: silver_v3={len(silver_rows)} "
            f"azure_parent={len(azure_rows)}"
        )

    commit = git_head(args.repo_root)
    config = {
        "layer_version": LAYER_VERSION,
        "built_on_spine": BUILT_ON_SPINE,
        "max_align_seconds": args.max_align_seconds,
        "honorific_equivalence": HONORIFIC_EQUIVALENCE,
        "mode": "production_internal_gate_validation_v1",
        "validation_schema": PRODUCTION_VALIDATION_SCHEMA,
        "pilot_segments": PILOT_SEGMENTS,
        "min_pilot_ar_recall": MIN_PILOT_AR_RECALL,
        "min_pilot_en_recall": MIN_PILOT_EN_RECALL,
        "min_tashkeel_density": MIN_TASHKEEL_DENSITY,
        "min_corroboration_share": MIN_CORROBORATION_SHARE,
        "max_immediate_duplicate_6grams": (
            MAX_IMMEDIATE_DUPLICATE_6GRAMS
        ),
        "required_views": REQUIRED_VIEWS,
        "anchor_views": sorted(ANCHOR_VIEWS),
    }
    config_hash = hashlib.sha256(
        json.dumps(
            config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    baseline_rows, _ = integrate_v3(
        silver_rows,
        azure_rows,
        args.max_align_seconds,
        commit,
        config_hash,
    )
    rows = [finalize_segment_row(row) for row in baseline_rows]
    provenance = provenance_rows(rows)

    rerun_baseline, _ = integrate_v3(
        silver_rows,
        azure_rows,
        args.max_align_seconds,
        commit,
        config_hash,
    )
    rerun_rows = [
        finalize_segment_row(row) for row in rerun_baseline
    ]
    deterministic = (
        hashlib.sha256(serialize(rows)).digest()
        == hashlib.sha256(serialize(rerun_rows)).digest()
    )

    spine_unchanged = all(
        row["silver_v3_text"] == baseline["silver_v3_text"]
        for row, baseline in zip(rows, baseline_rows)
    )
    validation = run_production_gates(
        rows,
        provenance,
        azure_rows,
        deterministic,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_path = (
        args.output_dir
        / f"{args.output_prefix}_segment_level.jsonl"
    )
    provenance_path = (
        args.output_dir
        / f"{args.output_prefix}_token_provenance.jsonl"
    )
    json_path = args.output_dir / f"{args.output_prefix}.json"
    manifest_path = args.output_dir / "PACKAGE_MANIFEST.json"
    report_path = (
        args.output_dir / f"{args.output_prefix}_run_report.json"
    )
    package_path = args.output_dir / f"{args.output_prefix}.zip"

    write_jsonl(segment_path, rows)
    write_jsonl(provenance_path, provenance)
    json_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "silver_plus_v4_fifth_view_export_v1"
                ),
                "layer_version": LAYER_VERSION,
                "built_on_spine": BUILT_ON_SPINE,
                "repository_commit": commit,
                "config_hash": config_hash,
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

    manifest = {
        "schema_version": (
            "silver_plus_v4_production_package_manifest_v1"
        ),
        "layer_version": LAYER_VERSION,
        "built_on_spine": BUILT_ON_SPINE,
        "repository_commit": commit,
        "config_hash": config_hash,
        "segment_count": len(rows),
        "evaluation_performed": True,
        "validation_performed": True,
        "validation": validation,
        "files": [],
    }
    for path in (segment_path, provenance_path, json_path):
        manifest["files"].append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = {
        "schema_version": "silver_plus_v4_production_run_report_v1",
        "layer_version": LAYER_VERSION,
        "built_on_spine": BUILT_ON_SPINE,
        "repository_commit": commit,
        "config_hash": config_hash,
        "segment_count": len(rows),
        "deterministic": deterministic,
        "silver_v3_text_byte_identical_to_spine": spine_unchanged,
        "evaluation_performed": True,
        "validation_performed": True,
        "validation": validation,
    }
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with zipfile.ZipFile(
        package_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in (
            segment_path,
            provenance_path,
            json_path,
            manifest_path,
            report_path,
        ):
            archive.write(path, arcname=path.name)

    completed = (
        deterministic
        and spine_unchanged
        and validation["all_gates_pass"] is True
    )
    result = {
        "completed": completed,
        "evaluation_performed": True,
        "validation_performed": True,
        "validation": validation,
        "segment_count": len(rows),
        "segment_jsonl": str(segment_path),
        "zip": str(package_path),
    }
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    if not deterministic:
        raise RuntimeError(
            "Silver+ v4 production rerun was not deterministic"
        )
    if not spine_unchanged:
        raise RuntimeError(
            "silver_v3_text changed during Silver+ v4 production "
            "integration"
        )
    if validation["all_gates_pass"] is not True:
        failed = [
            name
            for name, gate in validation["gates"].items()
            if gate.get("pass") is not True
        ]
        raise RuntimeError(
            "Silver+ v4 production validation failed: "
            + ", ".join(failed)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
