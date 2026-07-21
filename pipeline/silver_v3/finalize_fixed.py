from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SEGMENT_SCHEMA = "silver_v3_segment_level_v4"
MIN_A_TIER_RATIO = 0.60
MAX_IMMEDIATE_DUPLICATE_6GRAMS = 1
PILOT_SEGMENTS = 30
PILOT_MIN_A_TIER_RATIO = 0.75
PILOT_MAX_IMMEDIATE_DUPLICATE_6GRAMS = 1
DUPLICATE_REPAIR_SCHEMA = (
    "silver_v3_validator_semantic_duplicate_repair_v1"
)


def run_step(
    name: str,
    command: list[str],
    cwd: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    record: dict[str, Any] = {
        "name": name,
        "command": command,
        "returncode": result.returncode,
        "wall_seconds": round(
            time.perf_counter() - started,
            3,
        ),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
    }
    print("=" * 100, flush=True)
    print(name, flush=True)
    print("=" * 100, flush=True)
    print(result.stdout or "<empty stdout>", flush=True)
    if result.stderr:
        print("STDERR", flush=True)
        print(result.stderr, flush=True)
    return record


def require_success(record: dict[str, Any]) -> None:
    if record["returncode"] != 0:
        raise RuntimeError(
            f"{record['name']} failed with return code "
            f"{record['returncode']}"
        )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Invalid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(
            f"Expected JSON object: {path}"
        )
    return value


def validate_jsonl(
    path: Path,
    *,
    lecture_id: str | None = None,
) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)

    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL at "
                    f"{path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"Non-object JSONL row at "
                    f"{path}:{line_number}"
                )
            if (
                lecture_id is not None
                and value.get("lecture_id")
                not in {None, lecture_id}
            ):
                raise RuntimeError(
                    f"Lecture identity mismatch at "
                    f"{path}:{line_number}: "
                    f"{value.get('lecture_id')!r}"
                )
            count += 1

    if count == 0:
        raise RuntimeError(
            f"JSONL contains no rows: {path}"
        )
    return count


def immutable_title(lecture_id: str) -> str:
    number = lecture_id.rsplit("_", 1)[-1]
    return (
        f"Lecture {number} — "
        "Silver v3 Repaired Multiview Transcript"
    )


def validate_reconciliation_invariants(
    payload: dict[str, Any],
) -> dict[str, Any]:
    validation = payload.get("validation", {})
    if not isinstance(validation, dict):
        raise RuntimeError(
            "Reconciliation validation section is missing"
        )

    failures = {
        "segment_positions_ordered": (
            validation.get(
                "segment_positions_ordered"
            )
            is not True
        ),
        "segment_ids_unique": (
            validation.get("segment_ids_unique")
            is not True
        ),
        "chronology_error_count": (
            int(
                validation.get(
                    "chronology_error_count",
                    -1,
                )
            )
            != 0
        ),
        "zero_drop_invariant": (
            validation.get("zero_drop_invariant")
            is not True
        ),
        "unaccounted_observation_count": (
            int(
                validation.get(
                    "unaccounted_observation_count",
                    -1,
                )
            )
            != 0
        ),
    }
    failed = sorted(
        key
        for key, value in failures.items()
        if value
    )
    if failed:
        raise RuntimeError(
            "Reconciliation failed non-quality "
            "invariants: "
            + ", ".join(failed)
        )
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the immutable Lecture 001 "
            "Silver v3 repair/finalization from "
            "existing hosted-view outputs. The "
            "unchanged quality validator is the "
            "authoritative pass/fail gate."
        )
    )
    parser.add_argument(
        "--lecture-id",
        required=True,
    )
    parser.add_argument(
        "--silver-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    parser.add_argument("--output-prefix")
    parser.add_argument("--title")
    parser.add_argument(
        "--schema-version",
        default=SEGMENT_SCHEMA,
    )
    parser.add_argument(
        "--min-a-tier-ratio",
        type=float,
        default=MIN_A_TIER_RATIO,
    )
    parser.add_argument(
        "--max-repeated-6grams",
        type=int,
        default=MAX_IMMEDIATE_DUPLICATE_6GRAMS,
    )
    parser.add_argument(
        "--pilot-segments",
        type=int,
        default=PILOT_SEGMENTS,
    )
    parser.add_argument(
        "--pilot-min-a-tier-ratio",
        type=float,
        default=PILOT_MIN_A_TIER_RATIO,
    )
    parser.add_argument(
        "--pilot-max-repeated-6grams",
        type=int,
        default=(
            PILOT_MAX_IMMEDIATE_DUPLICATE_6GRAMS
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    silver_root = args.silver_root.resolve()
    expected_output_dir = (
        silver_root / "reconciled_fixed"
    ).resolve()
    output_dir = (
        args.output_dir
        or expected_output_dir
    ).resolve()
    expected_prefix = (
        f"{args.lecture_id}_silver_v3_fixed"
    )
    output_prefix = (
        args.output_prefix
        or expected_prefix
    )
    title = (
        args.title
        or immutable_title(args.lecture_id)
    )

    immutable_values = {
        "schema_version": (
            args.schema_version,
            SEGMENT_SCHEMA,
        ),
        "min_a_tier_ratio": (
            args.min_a_tier_ratio,
            MIN_A_TIER_RATIO,
        ),
        "max_immediate_duplicate_6grams": (
            args.max_repeated_6grams,
            MAX_IMMEDIATE_DUPLICATE_6GRAMS,
        ),
        "pilot_segments": (
            args.pilot_segments,
            PILOT_SEGMENTS,
        ),
        "pilot_min_a_tier_ratio": (
            args.pilot_min_a_tier_ratio,
            PILOT_MIN_A_TIER_RATIO,
        ),
        "pilot_max_immediate_duplicate_6grams": (
            args.pilot_max_repeated_6grams,
            PILOT_MAX_IMMEDIATE_DUPLICATE_6GRAMS,
        ),
    }
    mismatches = {
        key: {
            "observed": observed,
            "expected": expected,
        }
        for key, (
            observed,
            expected,
        ) in immutable_values.items()
        if observed != expected
    }
    if mismatches:
        raise ValueError(
            "Immutable Silver v3 contract values "
            "cannot be overridden: "
            + json.dumps(
                mismatches,
                sort_keys=True,
            )
        )
    if output_dir != expected_output_dir:
        raise ValueError(
            "Immutable repaired output directory "
            f"is {expected_output_dir}; "
            f"got {output_dir}"
        )
    if output_prefix != expected_prefix:
        raise ValueError(
            "Immutable repaired output prefix is "
            f"{expected_prefix}; "
            f"got {output_prefix}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    normalized_root = (
        silver_root / "normalized"
    )
    normalization_report = (
        silver_root
        / "silver_v3_normalization_report.json"
    )
    canonical_manifest = (
        silver_root
        / "manifests"
        / (
            f"{args.lecture_id}"
            "_canonical_manifest.jsonl"
        )
    )
    segment_jsonl = (
        output_dir
        / f"{output_prefix}_segment_level.jsonl"
    )
    provenance_jsonl = (
        output_dir
        / (
            f"{output_prefix}"
            "_token_provenance.jsonl"
        )
    )
    export_json = (
        output_dir / f"{output_prefix}.json"
    )
    docx_path = (
        output_dir / f"{output_prefix}.docx"
    )
    reconciliation_report = (
        output_dir
        / f"{output_prefix}_report.json"
    )
    duplicate_repair_report = (
        output_dir
        / (
            f"{output_prefix}"
            "_validator_duplicate_repair_report.json"
        )
    )
    quality_report = (
        output_dir
        / f"{output_prefix}_quality_report.json"
    )
    run_report = (
        output_dir
        / (
            f"{output_prefix}"
            "_finalization_report.json"
        )
    )

    required_inputs = [
        canonical_manifest,
        silver_root
        / "manifests"
        / f"{args.lecture_id}_whole.jsonl",
        silver_root
        / "manifests"
        / (
            f"{args.lecture_id}"
            "_canonical_20s.jsonl"
        ),
        silver_root
        / "manifests"
        / (
            f"{args.lecture_id}"
            "_context_10s_stride_5s.jsonl"
        ),
        silver_root
        / "manifests"
        / (
            f"{args.lecture_id}"
            "_local_2p5s_contiguous.jsonl"
        ),
    ]
    missing = [
        str(path)
        for path in required_inputs
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Silver v3 inputs:\n"
            + "\n".join(missing)
        )

    steps: list[dict[str, Any]] = []
    started = time.perf_counter()
    error: str | None = None
    passed = False
    reconciliation_nonzero_accepted = False
    duplicate_repair: dict[str, Any] | None = None

    try:
        normalize_record = run_step(
            (
                "1. Normalize hosted views "
                "with timing repair"
            ),
            [
                sys.executable,
                "-m",
                "pipeline.silver_v3.normalize_views",
                "--lecture-id",
                args.lecture_id,
                "--silver-root",
                str(silver_root),
                "--report",
                str(normalization_report),
            ],
            repo_root,
        )
        steps.append(normalize_record)
        require_success(normalize_record)

        normalization = read_json(
            normalization_report
        )
        if normalization.get("passed") is not True:
            raise RuntimeError(
                "Normalization report has passed=false"
            )

        reconcile_record = run_step(
            (
                "2. Reconcile repaired "
                "Silver v3 lattice"
            ),
            [
                sys.executable,
                str(
                    repo_root
                    / "scripts"
                    / "reconcile_silver_v3_lattice.py"
                ),
                "--lecture-id",
                args.lecture_id,
                "--whole",
                str(
                    normalized_root
                    / (
                        f"{args.lecture_id}"
                        "_whole_normalized.jsonl"
                    )
                ),
                "--canonical",
                str(
                    normalized_root
                    / (
                        f"{args.lecture_id}"
                        "_canonical_20s_normalized.jsonl"
                    )
                ),
                "--context",
                str(
                    normalized_root
                    / (
                        f"{args.lecture_id}"
                        "_context_10s_stride_5s"
                        "_normalized.jsonl"
                    )
                ),
                "--local",
                str(
                    normalized_root
                    / (
                        f"{args.lecture_id}"
                        "_local_2p5s_contiguous"
                        "_normalized.jsonl"
                    )
                ),
                "--canonical-manifest",
                str(canonical_manifest),
                "--output-dir",
                str(output_dir),
                "--output-prefix",
                output_prefix,
                "--schema-version",
                args.schema_version,
                "--title",
                title,
            ],
            repo_root,
        )
        steps.append(reconcile_record)

        required_outputs = [
            segment_jsonl,
            provenance_jsonl,
            export_json,
            docx_path,
            reconciliation_report,
        ]
        missing_outputs = [
            str(path)
            for path in required_outputs
            if (
                not path.is_file()
                or path.stat().st_size == 0
            )
        ]
        if missing_outputs:
            raise FileNotFoundError(
                "Reconciliation did not produce "
                "required outputs:\n"
                + "\n".join(missing_outputs)
            )

        validate_jsonl(
            segment_jsonl,
            lecture_id=args.lecture_id,
        )
        validate_jsonl(provenance_jsonl)
        export_payload = read_json(export_json)
        reconciliation_payload = read_json(
            reconciliation_report
        )
        if (
            export_payload.get("lecture_id")
            != args.lecture_id
        ):
            raise RuntimeError(
                "Repaired export lecture "
                "identity mismatch"
            )
        if (
            reconciliation_payload.get("lecture_id")
            != args.lecture_id
        ):
            raise RuntimeError(
                "Reconciliation report lecture "
                "identity mismatch"
            )

        reconciliation_validation = (
            validate_reconciliation_invariants(
                reconciliation_payload
            )
        )
        if reconcile_record["returncode"] != 0:
            reconciliation_nonzero_accepted = True
            reconcile_record[
                "nonzero_return_accepted_for_"
                "authoritative_quality_validation"
            ] = True
            reconcile_record[
                "diagnostic_validation"
            ] = reconciliation_validation

        duplicate_record = run_step(
            (
                "3. Repair validator-semantic "
                "immediate duplicate overlap artifacts"
            ),
            [
                sys.executable,
                "-m",
                (
                    "pipeline.silver_v3."
                    "repair_immediate_duplicates"
                ),
                "--segment-jsonl",
                str(segment_jsonl),
                "--token-provenance-jsonl",
                str(provenance_jsonl),
                "--export-json",
                str(export_json),
                "--docx",
                str(docx_path),
                "--reconciliation-report",
                str(reconciliation_report),
                "--title",
                title,
                "--maximum-remaining",
                str(
                    MAX_IMMEDIATE_DUPLICATE_6GRAMS
                ),
                "--output-report",
                str(duplicate_repair_report),
            ],
            repo_root,
        )
        steps.append(duplicate_record)
        require_success(duplicate_record)

        duplicate_repair = read_json(
            duplicate_repair_report
        )
        if (
            duplicate_repair.get("schema_version")
            != DUPLICATE_REPAIR_SCHEMA
        ):
            raise RuntimeError(
                "Unexpected duplicate repair schema"
            )
        if duplicate_repair.get("passed") is not True:
            raise RuntimeError(
                "Duplicate repair report has "
                "passed=false"
            )
        if (
            duplicate_repair.get(
                "evidence_accounting_preserved"
            )
            is not True
        ):
            raise RuntimeError(
                "Duplicate repair did not preserve "
                "evidence accounting"
            )

        post_repair_reconciliation = read_json(
            reconciliation_report
        )
        validate_reconciliation_invariants(
            post_repair_reconciliation
        )
        validate_jsonl(
            segment_jsonl,
            lecture_id=args.lecture_id,
        )
        validate_jsonl(provenance_jsonl)

        quality_record = run_step(
            (
                "4. Apply unchanged Silver v3 "
                "production quality gates"
            ),
            [
                sys.executable,
                "-m",
                (
                    "pipeline.silver_v3."
                    "validate_quality"
                ),
                "--segment-jsonl",
                str(segment_jsonl),
                "--normalization-report",
                str(normalization_report),
                "--output",
                str(quality_report),
                "--min-a-tier-ratio",
                str(MIN_A_TIER_RATIO),
                "--max-repeated-6grams",
                str(
                    MAX_IMMEDIATE_DUPLICATE_6GRAMS
                ),
                "--pilot-segments",
                str(PILOT_SEGMENTS),
                "--pilot-min-a-tier-ratio",
                str(PILOT_MIN_A_TIER_RATIO),
                "--pilot-max-repeated-6grams",
                str(
                    PILOT_MAX_IMMEDIATE_DUPLICATE_6GRAMS
                ),
            ],
            repo_root,
        )
        steps.append(quality_record)
        quality = read_json(quality_report)
        if (
            quality_record["returncode"] != 0
            or quality.get("passed") is not True
        ):
            raise RuntimeError(
                "Unchanged Silver v3 quality "
                "validator failed: "
                + json.dumps(
                    quality.get("gates", {}),
                    sort_keys=True,
                )
            )
        passed = True
    except Exception as exc:
        error = repr(exc)

    report = {
        "schema_version": (
            "silver_v3_fixed_"
            "finalization_report_v1"
        ),
        "contract_reference": (
            "lecture_001_silver_v3_"
            "repaired_export_package_v1"
        ),
        "lecture_id": args.lecture_id,
        "silver_root": str(silver_root),
        "output_dir": str(output_dir),
        "output_prefix": output_prefix,
        "segment_schema": SEGMENT_SCHEMA,
        "title": title,
        "thresholds": {
            "min_a_tier_ratio": (
                MIN_A_TIER_RATIO
            ),
            "max_immediate_duplicate_6grams": (
                MAX_IMMEDIATE_DUPLICATE_6GRAMS
            ),
            "pilot_segments": PILOT_SEGMENTS,
            "pilot_min_a_tier_ratio": (
                PILOT_MIN_A_TIER_RATIO
            ),
            (
                "pilot_max_immediate_"
                "duplicate_6grams"
            ): (
                PILOT_MAX_IMMEDIATE_DUPLICATE_6GRAMS
            ),
        },
        (
            "reconciliation_nonzero_accepted_"
            "for_authoritative_quality_validation"
        ): reconciliation_nonzero_accepted,
        "duplicate_repair": duplicate_repair,
        "wall_seconds": round(
            time.perf_counter() - started,
            3,
        ),
        "steps": steps,
        "normalization_report": str(
            normalization_report
        ),
        "reconciliation_report": str(
            reconciliation_report
        ),
        "duplicate_repair_report": str(
            duplicate_repair_report
        ),
        "quality_report": str(quality_report),
        "segment_jsonl": str(segment_jsonl),
        "error": error,
        "passed": passed,
    }
    run_report.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
