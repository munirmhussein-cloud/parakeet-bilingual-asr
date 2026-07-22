from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path
from typing import Any

STAMP_ORDER = (
    "*run_report*.json",
    "*PACKAGE_MANIFEST*.json",
    "*manifest*.json",
    "*report*.json",
)
EXPECTED_GATES = {
    "all_views_present",
    "corroboration_floor",
    "determinism",
    "duplicate_budget",
    "no_regression_vs_silver_v3",
    "pilot_window_union_reproduction",
    "vocalization_floor",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def unique_glob(package_dir: Path, pattern: str) -> Path:
    matches = sorted(package_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {pattern} in {package_dir}; "
            f"observed={[path.name for path in matches]}"
        )
    return matches[0]


def load_stamp(package_dir: Path) -> tuple[Path, dict[str, Any]]:
    for pattern in STAMP_ORDER:
        for path_text in sorted(
            glob.glob(str(package_dir / pattern))
        ):
            path = Path(path_text)
            value = load_json(path)
            if value.get("repository_commit"):
                return path, value
    raise RuntimeError("No JSON file carries repository_commit")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Silver+ v4 downstream-ingest stamp, "
            "empty-text contract, corroboration floor, and "
            "duplicate budget."
        )
    )
    parser.add_argument("package_dir", type=Path)
    args = parser.parse_args()

    package_dir = args.package_dir.resolve()
    if not package_dir.is_dir():
        raise FileNotFoundError(package_dir)

    failures: list[str] = []

    stamp_path, stamp = load_stamp(package_dir)
    validation = stamp.get("validation")
    all_gates_pass = (
        validation.get("all_gates_pass")
        if isinstance(validation, dict)
        else None
    )
    blocker_1 = all_gates_pass is True
    print(
        f"BLOCKER 1 stamp: {'PASS' if blocker_1 else 'FAIL'} "
        f"(reads {stamp_path.name}; "
        f"validation.all_gates_pass={all_gates_pass!r}; "
        f"config_hash={'ok' if stamp.get('config_hash') else 'MISSING'}; "
        f"repository_commit="
        f"{'ok' if stamp.get('repository_commit') else 'MISSING'})"
    )
    if not blocker_1:
        failures.append("validation.all_gates_pass is not true")

    if not isinstance(validation, dict):
        failures.append("nested validation object is missing")
    else:
        gates = validation.get("gates")
        if not isinstance(gates, dict) or set(gates) != EXPECTED_GATES:
            failures.append(
                "validation gate set differs: "
                f"observed="
                f"{sorted(gates) if isinstance(gates, dict) else gates!r}"
            )
        elif any(
            not isinstance(gate, dict)
            or gate.get("pass") is not True
            for gate in gates.values()
        ):
            failures.append("one or more nested validation gates failed")
        if validation.get("validation_performed") is not True:
            failures.append(
                "validation.validation_performed is not true"
            )

    report_path = unique_glob(package_dir, "*run_report*.json")
    manifest_path = package_dir / "PACKAGE_MANIFEST.json"
    if manifest_path.is_file():
        report = load_json(report_path)
        manifest = load_json(manifest_path)
        if manifest.get("validation") != report.get("validation"):
            failures.append(
                "PACKAGE_MANIFEST validation does not mirror run_report"
            )

    segment_path = unique_glob(
        package_dir,
        "*segment_level*.jsonl",
    )
    unflagged_empty: list[str] = []
    duplicate_sum = 0
    with segment_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(
                row.get("silver_plus_v4_text") or ""
            ).strip()
            segment_id = str(
                row.get("segment_id")
                or row.get("embedded_seg_id")
                or f"line_{line_number}"
            )
            if (
                not text
                and row.get("has_silver_plus_v4_text") is not False
            ):
                unflagged_empty.append(segment_id)
            if (
                text
                and "has_silver_plus_v4_text" in row
                and row.get("has_silver_plus_v4_text") is not True
            ):
                failures.append(
                    "non-empty segment is not flagged true: "
                    + segment_id
                )
            duplicate_sum += int(
                row.get("immediate_duplicate_6gram_count", 0) or 0
            )

    blocker_2 = not unflagged_empty
    print(
        f"BLOCKER 2 empty-text: {'PASS' if blocker_2 else 'FAIL'} "
        f"(unflagged empty segments: "
        f"{unflagged_empty if unflagged_empty else 'none'})"
    )
    if not blocker_2:
        failures.append(
            "unflagged empty segments: "
            + ", ".join(unflagged_empty)
        )

    provenance_path = unique_glob(
        package_dir,
        "*token_provenance*.jsonl",
    )
    tiers: collections.Counter[str | None] = collections.Counter()
    with provenance_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tiers[
                row.get("acceptance_tier") or row.get("tier")
            ] += 1
    total = sum(tiers.values())
    corroborated = tiers.get("A_corroborated", 0)
    corroboration = corroborated / total if total else 0.0
    corroboration_pass = corroboration >= 0.60
    duplicate_pass = duplicate_sum <= 5
    print(
        f"corroboration: {corroborated}/{total} = "
        f"{corroboration:.2f} "
        f"({'PASS' if corroboration_pass else 'FAIL <0.60'})"
    )
    print(
        f"dup6 sum: {duplicate_sum} "
        f"({'PASS' if duplicate_pass else 'FAIL >5'})"
    )
    if not corroboration_pass:
        failures.append("corroboration share is below 0.60")
    if not duplicate_pass:
        failures.append(
            "summed immediate duplicate six-grams exceed 5"
        )

    if failures:
        print("\nDO NOT PROCEED — fix and re-seal")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
