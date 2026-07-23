from __future__ import annotations

"""Produce one clean-room Silver+ v4 lecture package.

The authoritative package is written to:

    production_a100/lectures/lecture_NNN/
        batch_silver_plus_v4/fifth_view_integrated/

That directory contains exactly four registrable files: segment JSONL, token
provenance JSONL, run report JSON, and PACKAGE_MANIFEST.json. The legacy
silver_plus_v4 tree is never modified.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_GATES = {
    "all_views_present",
    "corroboration_floor",
    "determinism",
    "duplicate_budget",
    "no_regression_vs_silver_v3",
    "pilot_window_union_reproduction",
    "vocalization_floor",
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def git_capture(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def normalize_lecture(value: str) -> str:
    match = re.search(r"(\d{1,3})", str(value))
    if not match:
        raise ValueError(f"Invalid lecture value: {value}")
    number = int(match.group(1))
    if not 1 <= number <= 104:
        raise ValueError("Lecture must be between 1 and 104")
    return f"lecture_{number:03d}"


def resolve_silver_v3_segment(silver_root: Path) -> Path:
    marker_path = silver_root / ".silver_v3_complete.json"
    if marker_path.is_file():
        marker = load_json(marker_path)
        candidate = Path(str(marker.get("segment_jsonl") or ""))
        if marker.get("status") == "completed" and candidate.is_file():
            return candidate

    candidates = sorted(
        {
            *silver_root.glob(
                "repaired_package/*/*_silver_v3_fixed_segment_level.jsonl"
            ),
            *silver_root.glob(
                "reconciled_fixed/*_silver_v3_fixed_segment_level.jsonl"
            ),
        }
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "Unable to resolve one Silver v3 segment spine; "
            f"observed={[str(path) for path in candidates]}"
        )
    return candidates[0]


def stream(command: list[str], *, cwd: Path) -> int:
    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def validate_package(package_dir: Path, lecture_id: str) -> dict[str, Any]:
    prefix = f"{lecture_id}_silver_plus_v4"
    expected = {
        f"{prefix}_segment_level.jsonl",
        f"{prefix}_token_provenance.jsonl",
        f"{prefix}_run_report.json",
        "PACKAGE_MANIFEST.json",
    }
    observed = {path.name for path in package_dir.iterdir() if path.is_file()}
    if observed != expected:
        raise RuntimeError(
            f"Four-file contract mismatch: expected={sorted(expected)} "
            f"observed={sorted(observed)}"
        )

    report_path = package_dir / f"{prefix}_run_report.json"
    manifest_path = package_dir / "PACKAGE_MANIFEST.json"
    segment_path = package_dir / f"{prefix}_segment_level.jsonl"
    provenance_path = package_dir / f"{prefix}_token_provenance.jsonl"

    report = load_json(report_path)
    manifest = load_json(manifest_path)
    validation = report.get("validation")
    if not isinstance(validation, dict):
        raise RuntimeError("Run report lacks nested validation object")
    if validation.get("all_gates_pass") is not True:
        raise RuntimeError("validation.all_gates_pass is not true")
    gates = validation.get("gates")
    if not isinstance(gates, dict) or set(gates) != EXPECTED_GATES:
        raise RuntimeError("Run report carries an unexpected validation gate set")
    failed = [
        name
        for name, gate in gates.items()
        if not isinstance(gate, dict) or gate.get("pass") is not True
    ]
    if failed:
        raise RuntimeError("Failed production gates: " + ", ".join(failed))
    if manifest.get("validation") != validation:
        raise RuntimeError("Manifest validation does not mirror run report")

    segment_count = 0
    duplicate_sum = 0
    unflagged_empty: list[str] = []
    with segment_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            segment_count += 1
            text = str(row.get("silver_plus_v4_text") or "").strip()
            segment_id = str(
                row.get("segment_id")
                or row.get("embedded_seg_id")
                or f"line_{line_number}"
            )
            if not text and row.get("has_silver_plus_v4_text") is not False:
                unflagged_empty.append(segment_id)
            if text and row.get("has_silver_plus_v4_text") is not True:
                raise RuntimeError(
                    f"Non-empty segment lacks true text flag: {segment_id}"
                )
            duplicate_sum += int(
                row.get("immediate_duplicate_6gram_count", 0) or 0
            )
    if unflagged_empty:
        raise RuntimeError(
            "Unflagged empty segments: " + ", ".join(unflagged_empty)
        )
    if duplicate_sum > 5:
        raise RuntimeError(f"Duplicate six-gram sum exceeds 5: {duplicate_sum}")

    total_tokens = 0
    corroborated_tokens = 0
    with provenance_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            total_tokens += 1
            if (
                row.get("acceptance_tier") or row.get("tier")
            ) == "A_corroborated":
                corroborated_tokens += 1
    corroboration = (
        corroborated_tokens / total_tokens if total_tokens else 0.0
    )
    if corroboration < 0.60:
        raise RuntimeError(
            f"Corroboration share is below 0.60: {corroboration:.6f}"
        )
    if report.get("segment_count") != segment_count:
        raise RuntimeError(
            "Run-report segment count does not match segment JSONL"
        )

    return {
        "repository_commit": report.get("repository_commit"),
        "config_hash": report.get("config_hash"),
        "segment_count": segment_count,
        "total_tokens": total_tokens,
        "corroborated_tokens": corroborated_tokens,
        "corroboration_share": corroboration,
        "duplicate_sum": duplicate_sum,
        "tashkeel_density": gates["vocalization_floor"].get(
            "tashkeel_density"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Produce one four-file batch_silver_plus_v4 package."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/content/parakeet-bilingual-asr"),
    )
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/parakeet-bilingual-asr"),
    )
    parser.add_argument("--lecture", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    drive_root = args.drive_root.resolve()
    lecture_id = normalize_lecture(args.lecture)
    lecture_root = (
        drive_root / "production_a100" / "lectures" / lecture_id
    )
    silver_root = lecture_root / "silver_v3"
    azure_parent = (
        lecture_root
        / "bronze_v3"
        / f"{lecture_id}_bronze_v3_azure_parent.jsonl"
    )
    producer = (
        repo_root
        / "scripts"
        / "run_batch_silver_plus_v4_production_lecture.py"
    )
    validator = (
        repo_root / "scripts" / "validate_silver_plus_v4_acceptance.py"
    )

    for required in (
        repo_root / ".git",
        lecture_root,
        silver_root,
        azure_parent,
        producer,
        validator,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    status = git_capture(
        repo_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    substantive = [
        line
        for line in status.splitlines()
        if "__pycache__" not in line and not line.rstrip().endswith(".pyc")
    ]
    if substantive:
        raise RuntimeError(
            "Repository working tree must be clean:\n" + "\n".join(substantive)
        )
    commit = git_capture(repo_root, "rev-parse", "HEAD")
    segment_spine = resolve_silver_v3_segment(silver_root)

    namespace_root = lecture_root / "batch_silver_plus_v4"
    authoritative = namespace_root / "fifth_view_integrated"
    marker_path = namespace_root / ".batch_silver_plus_v4_complete.json"

    if authoritative.is_dir() and not args.overwrite:
        report = validate_package(authoritative, lecture_id)
        if report.get("repository_commit") == commit:
            print(
                json.dumps(
                    {
                        "status": "skipped",
                        "reason": "current package is contract-valid",
                        "lecture_id": lecture_id,
                        "package_dir": str(authoritative),
                        **report,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        raise RuntimeError(
            "A valid package exists under another commit; rerun with --overwrite"
        )

    run_id = f"{utc_stamp()}_{commit[:8]}"
    stage = namespace_root / ".staging" / run_id
    candidate = namespace_root / ".candidate" / run_id
    stage.mkdir(parents=True, exist_ok=False)
    candidate.mkdir(parents=True, exist_ok=False)
    prefix = f"{lecture_id}_silver_plus_v4"

    command = [
        sys.executable,
        "-u",
        str(producer),
        "--repo-root",
        str(repo_root),
        "--silver-v3",
        str(segment_spine),
        "--azure-parent",
        str(azure_parent),
        "--output-dir",
        str(stage),
        "--output-prefix",
        prefix,
    ]
    return_code = stream(command, cwd=repo_root)
    if return_code != 0:
        raise RuntimeError(
            f"Silver+ v4 producer failed with return code {return_code}; "
            f"staging preserved at {stage}"
        )

    source_names = {
        "segment": f"{prefix}_segment_level.jsonl",
        "provenance": f"{prefix}_token_provenance.jsonl",
        "report": f"{prefix}_run_report.json",
    }
    for name in source_names.values():
        source = stage / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, candidate / name)

    source_manifest = load_json(stage / "PACKAGE_MANIFEST.json")
    report = load_json(candidate / source_names["report"])
    manifest = {
        **source_manifest,
        "schema_version": "batch_silver_plus_v4_package_manifest_v1",
        "namespace": "batch_silver_plus_v4",
        "package_contract": "four_registrable_files_v1",
        "repository_commit": commit,
        "validation": report.get("validation"),
        "files": [],
    }
    for name in source_names.values():
        path = candidate / name
        manifest["files"].append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    (candidate / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    validation_report = validate_package(candidate, lecture_id)
    validator_code = stream(
        [sys.executable, "-u", str(validator), str(candidate)],
        cwd=repo_root,
    )
    if validator_code != 0:
        raise RuntimeError(
            f"Acceptance validator failed; candidate preserved at {candidate}"
        )

    if authoritative.exists():
        superseded = namespace_root / "superseded" / run_id
        superseded.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(authoritative), str(superseded))
    authoritative.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(candidate), str(authoritative))
    shutil.rmtree(stage, ignore_errors=True)

    marker = {
        "schema_version": "batch_silver_plus_v4_completion_marker_v1",
        "status": "completed",
        "lecture_id": lecture_id,
        "repository_commit": commit,
        "silver_v3_segment_jsonl": str(segment_spine),
        "silver_v3_segment_sha256": sha256_file(segment_spine),
        "azure_parent_jsonl": str(azure_parent),
        "azure_parent_sha256": sha256_file(azure_parent),
        "package_dir": str(authoritative),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **validation_report,
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "completed",
                "lecture_id": lecture_id,
                "package_dir": str(authoritative),
                **validation_report,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
