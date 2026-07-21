from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = "silver_v3_repaired_export_package_v1"
QUALITY_SCHEMA = "silver_v3_quality_validation_v2"
REQUIRED_PACKAGE_FILES = {
    "PACKAGE_MANIFEST.json",
    "SHA256SUMS.txt",
    "silver_v3_normalization_report.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalize_lecture(value: str) -> str:
    match = re.search(r"(\d{1,3})", value)
    if not match:
        raise ValueError(f"Invalid lecture identifier: {value}")
    number = int(match.group(1))
    if not 1 <= number <= 104:
        raise ValueError(f"Lecture outside 001-104: {value}")
    return f"lecture_{number:03d}"


def run_streamed(command: list[str], cwd: Path, prefix: str) -> int:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        if text:
            print(f"[{prefix}] {text}", flush=True)
    return process.wait()


def validate_repaired_archive(
    archive_path: Path,
    lecture_id: str,
) -> dict[str, Any]:
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise RuntimeError(f"Missing repaired Silver v3 archive: {archive_path}")

    with zipfile.ZipFile(archive_path, "r") as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Corrupt ZIP member: {corrupt}")
        names = archive.namelist()
        roots = {name.split("/", 1)[0] for name in names if "/" in name}
        if len(roots) != 1:
            raise RuntimeError(f"Expected exactly one package root in {archive_path}")
        root = next(iter(roots))
        basename_map = {Path(name).name: name for name in names if not name.endswith("/")}
        output_prefix = f"{lecture_id}_silver_v3_fixed"
        required = REQUIRED_PACKAGE_FILES | {
            f"{output_prefix}.docx",
            f"{output_prefix}.json",
            f"{output_prefix}_quality_report.json",
            f"{output_prefix}_segment_level.jsonl",
            f"{output_prefix}_token_provenance.jsonl",
        }
        missing = sorted(required - set(basename_map))
        if missing:
            raise RuntimeError(f"Repaired archive missing required files: {missing}")

        manifest = json.loads(archive.read(basename_map["PACKAGE_MANIFEST.json"]))
        if manifest.get("schema_version") != CONTRACT_SCHEMA:
            raise RuntimeError("Repaired package schema mismatch")
        if manifest.get("lecture_id") != lecture_id:
            raise RuntimeError("Repaired package lecture mismatch")
        if manifest.get("quality_validation_schema") != QUALITY_SCHEMA:
            raise RuntimeError("Quality validation schema mismatch")
        if manifest.get("quality_passed") is not True:
            raise RuntimeError("Repaired package quality_passed is not true")

        manifest_files = manifest.get("files")
        if not isinstance(manifest_files, list) or len(manifest_files) != 6:
            raise RuntimeError("Repaired package manifest must list six source files")

        for item in manifest_files:
            filename = item.get("filename")
            if filename not in basename_map:
                raise RuntimeError(f"Manifest member missing from ZIP: {filename}")
            payload = archive.read(basename_map[filename])
            digest = hashlib.sha256(payload).hexdigest()
            if item.get("size_bytes") != len(payload):
                raise RuntimeError(f"Manifest size mismatch: {filename}")
            if item.get("sha256") != digest:
                raise RuntimeError(f"Manifest SHA mismatch: {filename}")

        checksum_lines = archive.read(basename_map["SHA256SUMS.txt"]).decode("utf-8").splitlines()
        checksums = {}
        for line in checksum_lines:
            if not line.strip():
                continue
            digest, filename = line.split(None, 1)
            checksums[filename.strip()] = digest
        for filename in required - {"SHA256SUMS.txt"}:
            if filename not in checksums:
                raise RuntimeError(f"Checksum entry missing: {filename}")
            payload = archive.read(basename_map[filename])
            if checksums[filename] != hashlib.sha256(payload).hexdigest():
                raise RuntimeError(f"Checksum mismatch: {filename}")

    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    if not checksum_path.is_file():
        raise RuntimeError(f"Missing archive checksum file: {checksum_path}")
    expected_archive_sha = checksum_path.read_text(encoding="utf-8").split()[0]
    actual_archive_sha = sha256_file(archive_path)
    if expected_archive_sha != actual_archive_sha:
        raise RuntimeError("Archive checksum mismatch")

    return {
        "archive": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": actual_archive_sha,
        "package_root": root,
        "zip_member_count": len(names),
        "manifest": manifest,
    }


def find_valid_repaired_archive(silver_root: Path, lecture_id: str) -> tuple[Path | None, dict[str, Any] | None]:
    final_package = silver_root / "final_package"
    candidates = sorted(
        final_package.glob(f"{lecture_id}_silver_v3_repaired_*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if final_package.is_dir() else []
    for candidate in candidates:
        try:
            return candidate, validate_repaired_archive(candidate, lecture_id)
        except Exception:
            continue
    return None, None


def ensure_quality_and_package(
    repo_root: Path,
    silver_root: Path,
    lecture_id: str,
) -> dict[str, Any]:
    output_prefix = f"{lecture_id}_silver_v3_fixed"
    reconciled = silver_root / "reconciled_fixed"
    segment_jsonl = reconciled / f"{output_prefix}_segment_level.jsonl"
    normalization_report = silver_root / "silver_v3_normalization_report.json"
    quality_report = reconciled / f"{output_prefix}_quality_report.json"

    required = [
        segment_jsonl,
        normalization_report,
        reconciled / f"{output_prefix}_token_provenance.jsonl",
        reconciled / f"{output_prefix}.json",
        reconciled / f"{output_prefix}.docx",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing repaired Silver outputs:\n" + "\n".join(missing))

    quality_command = [
        sys.executable,
        "-u",
        str(repo_root / "pipeline/silver_v3/validate_quality.py"),
        "--segment-jsonl",
        str(segment_jsonl),
        "--normalization-report",
        str(normalization_report),
        "--output",
        str(quality_report),
    ]
    quality_return = run_streamed(quality_command, repo_root, f"{lecture_id}:quality")
    if quality_return != 0:
        raise RuntimeError(f"{lecture_id} failed immutable Lecture 001 quality gates")

    quality = json.loads(quality_report.read_text(encoding="utf-8"))
    if quality.get("schema_version") != QUALITY_SCHEMA or quality.get("passed") is not True:
        raise RuntimeError(f"{lecture_id} quality report is not contract-valid")

    export_dir = silver_root / "repaired_package"
    final_package = silver_root / "final_package"
    package_command = [
        sys.executable,
        "-u",
        str(repo_root / "scripts/package_silver_v3_repaired.py"),
        "--repo-root",
        str(repo_root),
        "--silver-root",
        str(silver_root),
        "--lecture-id",
        lecture_id,
        "--output-prefix",
        output_prefix,
        "--export-dir",
        str(export_dir),
        "--copy-to",
        str(final_package),
    ]
    package_return = run_streamed(package_command, repo_root, f"{lecture_id}:package")
    if package_return != 0:
        raise RuntimeError(f"{lecture_id} repaired package creation failed")

    archive, report = find_valid_repaired_archive(silver_root, lecture_id)
    if archive is None or report is None:
        raise RuntimeError(f"{lecture_id} repaired archive did not validate")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Silver v3 sequentially and accept completion only when the immutable Lecture 001 repaired package contract validates."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("/content/parakeet-bilingual-asr"))
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/parakeet-bilingual-asr"))
    parser.add_argument("--start", default="lecture_001")
    parser.add_argument("--end", default="lecture_104")
    parser.add_argument("--lectures", nargs="*")
    parser.add_argument("--package-existing-only", action="store_true")
    parser.add_argument("--continue-after-failure", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    drive_root = args.drive_root.resolve()
    lectures_root = drive_root / "production_a100/lectures"
    state_path = drive_root / "production_a100/pipeline_state/silver_v3_contract_sequential_summary.json"
    batch_runner = repo_root / "scripts/run_silver_v3_batch_adaptive.py"
    if not batch_runner.is_file():
        batch_runner = repo_root / "scripts/run_silver_v3_batch.py"
    if not batch_runner.is_file():
        raise FileNotFoundError("No Silver v3 batch runner found")

    if args.lectures:
        lecture_ids = sorted({normalize_lecture(value) for value in args.lectures})
    else:
        start = int(normalize_lecture(args.start)[-3:])
        end = int(normalize_lecture(args.end)[-3:])
        if start > end:
            raise ValueError("--start cannot exceed --end")
        lecture_ids = [f"lecture_{number:03d}" for number in range(start, end + 1)]

    state: dict[str, Any] = {
        "schema_version": "silver_v3_contract_sequential_summary_v1",
        "contract_schema": CONTRACT_SCHEMA,
        "quality_schema": QUALITY_SCHEMA,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "results": {},
    }
    atomic_write_json(state_path, state)

    print("=" * 96)
    print("SILVER v3 IMMUTABLE CONTRACT SEQUENTIAL RUN")
    print("=" * 96)
    print("Lectures:", f"{lecture_ids[0]}-{lecture_ids[-1]}")
    print("Package existing only:", args.package_existing_only)
    print("State:", state_path)

    failures = 0
    for index, lecture_id in enumerate(lecture_ids, start=1):
        silver_root = lectures_root / lecture_id / "silver_v3"
        archive, report = find_valid_repaired_archive(silver_root, lecture_id)
        action = "skip" if archive else ("package" if args.package_existing_only else "run")
        print(f"{index:>3}/{len(lecture_ids)} {lecture_id}: {action.upper()}")
        if args.plan_only:
            continue
        if archive and report:
            state["results"][lecture_id] = {"status": "skipped_contract_valid", **report}
            state["updated_at"] = utc_now()
            atomic_write_json(state_path, state)
            continue

        started = time.monotonic()
        try:
            if not args.package_existing_only:
                command = [sys.executable, "-u", str(batch_runner), "--lectures", lecture_id]
                return_code = run_streamed(command, repo_root, lecture_id)
                if return_code != 0:
                    raise RuntimeError(f"Underlying Silver v3 runner failed with {return_code}")

            report = ensure_quality_and_package(repo_root, silver_root, lecture_id)
            state["results"][lecture_id] = {
                "status": "completed_contract_valid",
                "wall_seconds": round(time.monotonic() - started, 3),
                **report,
            }
            print(f"COMPLETED {lecture_id}: repaired contract valid", flush=True)
        except Exception as exc:
            failures += 1
            state["results"][lecture_id] = {
                "status": "failed",
                "wall_seconds": round(time.monotonic() - started, 3),
                "error": str(exc),
            }
            print(f"FAILED {lecture_id}: {exc}", flush=True)
            if not args.continue_after_failure:
                state["updated_at"] = utc_now()
                atomic_write_json(state_path, state)
                break
        state["updated_at"] = utc_now()
        atomic_write_json(state_path, state)

    state["finished_at"] = utc_now()
    state["failure_count"] = failures
    state["completed"] = failures == 0 and len(state["results"]) == len(lecture_ids)
    atomic_write_json(state_path, state)
    return 0 if state["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
