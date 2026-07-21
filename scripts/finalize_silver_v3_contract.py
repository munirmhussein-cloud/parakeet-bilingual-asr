from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "silver_v3_repaired_export_package_v1"
EXPECTED_QUALITY_SCHEMA = "silver_v3_quality_validation_v2"
EXPECTED_SEGMENT_SCHEMA = "silver_v3_segment_level_v4"
EXPECTED_EXPORT_SCHEMA = "silver_v3_segment_level_v4_export"
EXPECTED_PROVENANCE_SCHEMA = "silver_v3_segment_level_v4_token_provenance"
EXPECTED_NORMALIZATION_SCHEMA = "silver_v3_normalization_report_v4"
REQUIRED_VIEWS = (
    "whole",
    "canonical_20s",
    "context_10s_stride_5s",
    "local_2p5s_contiguous",
)
EXPECTED_THRESHOLDS = {
    "min_a_tier_ratio": 0.60,
    "max_immediate_duplicate_6grams": 1,
    "pilot_segments": 30,
    "pilot_min_a_tier_ratio": 0.75,
    "pilot_max_immediate_duplicate_6grams": 1,
    "require_anchor_contribution": True,
}
EXPECTED_GATES = {
    "normalization_passed",
    "required_input_views_pass",
    "full_a_tier_ratio_pass",
    "full_immediate_duplicate_6grams_pass",
    "pilot_a_tier_ratio_pass",
    "pilot_immediate_duplicate_6grams_pass",
    "anchor_contribution_pass",
    "segments_present",
    "tokens_present",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def json_from_bytes(payload: bytes, filename: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid JSON payload: {filename}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {filename}")
    return value


def jsonl_from_bytes(payload: bytes, filename: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Invalid UTF-8 JSONL: {filename}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSONL at {filename}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Non-object JSONL row at {filename}:{line_number}")
        rows.append(value)
    if not rows:
        raise RuntimeError(f"JSONL contains no rows: {filename}")
    return rows


def run(command: list[str], cwd: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout[-8000:]}\n"
            f"stderr:\n{result.stderr[-8000:]}"
        )
    output = result.stdout.strip()
    try:
        return json.loads(output) if output else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Command did not emit a JSON object: {' '.join(command)}\n"
            f"stdout:\n{result.stdout[-8000:]}"
        ) from exc


def source_filenames(lecture_id: str) -> tuple[str, ...]:
    prefix = f"{lecture_id}_silver_v3_fixed"
    return (
        f"{prefix}_segment_level.jsonl",
        f"{prefix}_token_provenance.jsonl",
        f"{prefix}.json",
        f"{prefix}.docx",
        f"{prefix}_quality_report.json",
        "silver_v3_normalization_report.json",
    )


def package_filenames(lecture_id: str) -> set[str]:
    return {
        "PACKAGE_MANIFEST.json",
        "SHA256SUMS.txt",
        *source_filenames(lecture_id),
    }


def parse_checksum_payload(payload: bytes) -> dict[str, str]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"SHA256SUMS.txt is not UTF-8: {exc}") from exc
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if "  " not in line:
            raise RuntimeError(f"Malformed checksum line {line_number}: {line!r}")
        digest, filename = line.split("  ", 1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"Malformed SHA-256 on line {line_number}")
        if not filename or filename in checksums:
            raise RuntimeError(f"Duplicate or empty checksum filename: {filename!r}")
        checksums[filename] = digest
    return checksums


def _equal_number(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-12
    return left == right


def validate_thresholds(quality: dict[str, Any]) -> None:
    thresholds = quality.get("thresholds")
    if not isinstance(thresholds, dict):
        raise RuntimeError("Quality report thresholds are missing")
    if set(thresholds) != set(EXPECTED_THRESHOLDS):
        raise RuntimeError(
            "Quality threshold keys differ from the immutable contract: "
            f"observed={sorted(thresholds)} expected={sorted(EXPECTED_THRESHOLDS)}"
        )
    mismatches = {
        key: {"observed": thresholds.get(key), "expected": expected}
        for key, expected in EXPECTED_THRESHOLDS.items()
        if not _equal_number(thresholds.get(key), expected)
    }
    if mismatches:
        raise RuntimeError(
            "Quality thresholds differ from the immutable contract: "
            + json.dumps(mismatches, sort_keys=True)
        )


def validate_payload_contract(
    *,
    lecture_id: str,
    package_root: str,
    manifest: dict[str, Any],
    payloads: dict[str, bytes],
    expected_repository_commit: str | None,
) -> dict[str, Any]:
    expected_sources = set(source_filenames(lecture_id))
    if manifest.get("schema_version") != EXPECTED_SCHEMA:
        raise RuntimeError(f"Unexpected package schema: {manifest.get('schema_version')!r}")
    if manifest.get("lecture_id") != lecture_id:
        raise RuntimeError("Package lecture ID mismatch")
    expected_prefix = f"{lecture_id}_silver_v3_fixed"
    if manifest.get("output_prefix") != expected_prefix:
        raise RuntimeError("Package output_prefix mismatch")
    if manifest.get("quality_validation_schema") != EXPECTED_QUALITY_SCHEMA:
        raise RuntimeError(
            f"Unexpected quality schema: {manifest.get('quality_validation_schema')!r}"
        )
    if manifest.get("quality_passed") is not True:
        raise RuntimeError("Package quality_passed is not true")

    repository_commit = str(manifest.get("repository_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", repository_commit):
        raise RuntimeError("Package repository_commit is not a full Git commit")
    if expected_repository_commit is not None and repository_commit != expected_repository_commit:
        raise RuntimeError(
            f"Package commit mismatch: {repository_commit} != {expected_repository_commit}"
        )
    expected_root = f"{lecture_id}_silver_v3_repaired_{repository_commit[:8]}"
    if package_root != expected_root:
        raise RuntimeError(f"Package root mismatch: {package_root!r} != {expected_root!r}")

    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 6:
        raise RuntimeError("Package manifest must list exactly six source files")
    manifest_by_name: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError("Package manifest file entry is not an object")
        filename = item.get("filename")
        if not isinstance(filename, str) or filename in manifest_by_name:
            raise RuntimeError(f"Invalid or duplicate manifest filename: {filename!r}")
        manifest_by_name[filename] = item
    if set(manifest_by_name) != expected_sources:
        raise RuntimeError(
            "Package source-file contract mismatch. "
            f"expected={sorted(expected_sources)} observed={sorted(manifest_by_name)}"
        )

    for filename, item in manifest_by_name.items():
        payload = payloads.get(filename)
        if payload is None:
            raise RuntimeError(f"Package payload missing: {filename}")
        digest = sha256_bytes(payload)
        if item.get("sha256") != digest:
            raise RuntimeError(f"Manifest SHA mismatch for {filename}")
        if item.get("size_bytes") != len(payload):
            raise RuntimeError(f"Manifest size mismatch for {filename}")
        source_path = str(item.get("source_path", ""))
        if Path(source_path).name != filename:
            raise RuntimeError(f"Manifest source_path filename mismatch for {filename}")
        if lecture_id not in source_path:
            raise RuntimeError(f"Manifest source_path lecture mismatch for {filename}")

    prefix = f"{lecture_id}_silver_v3_fixed"
    segment_name = f"{prefix}_segment_level.jsonl"
    provenance_name = f"{prefix}_token_provenance.jsonl"
    export_name = f"{prefix}.json"
    quality_name = f"{prefix}_quality_report.json"
    normalization_name = "silver_v3_normalization_report.json"

    segment_rows = jsonl_from_bytes(payloads[segment_name], segment_name)
    provenance_rows = jsonl_from_bytes(payloads[provenance_name], provenance_name)
    export_payload = json_from_bytes(payloads[export_name], export_name)
    quality = json_from_bytes(payloads[quality_name], quality_name)
    normalization = json_from_bytes(payloads[normalization_name], normalization_name)

    if export_payload.get("schema_version") != EXPECTED_EXPORT_SCHEMA:
        raise RuntimeError("Fixed JSON export schema mismatch")
    if export_payload.get("lecture_id") != lecture_id:
        raise RuntimeError("Fixed JSON export lecture mismatch")
    if export_payload.get("segment_count") != len(segment_rows):
        raise RuntimeError("Fixed JSON export segment_count mismatch")
    if export_payload.get("segments") != segment_rows:
        raise RuntimeError("Fixed JSON export segments differ from segment JSONL")

    positions: list[int] = []
    segment_ids: list[str] = []
    nested_provenance: list[tuple[int, str, int, str]] = []
    total_tokens = 0
    for row_number, row in enumerate(segment_rows, start=1):
        if row.get("schema_version") != EXPECTED_SEGMENT_SCHEMA:
            raise RuntimeError(f"Segment schema mismatch at row {row_number}")
        if row.get("lecture_id") != lecture_id:
            raise RuntimeError(f"Segment lecture mismatch at row {row_number}")
        position = row.get("segment_position")
        if not isinstance(position, int):
            raise RuntimeError(f"Segment position is not an integer at row {row_number}")
        if row.get("segment_index") != position:
            raise RuntimeError(f"Segment index/position mismatch at row {row_number}")
        segment_id = str(row.get("segment_id", ""))
        if not segment_id.startswith(f"{lecture_id}__canonical_20s__"):
            raise RuntimeError(f"Segment ID lecture/view mismatch at row {row_number}")
        tokens = row.get("tokens")
        if not isinstance(tokens, list) or row.get("token_count") != len(tokens):
            raise RuntimeError(f"Segment token_count mismatch at row {row_number}")
        positions.append(position)
        segment_ids.append(segment_id)
        total_tokens += len(tokens)
        for token in tokens:
            if not isinstance(token, dict) or not isinstance(token.get("slot_id"), int):
                raise RuntimeError(f"Malformed token in segment row {row_number}")
            nested_provenance.append(
                (
                    position,
                    segment_id,
                    int(token["slot_id"]),
                    str(token.get("text", "")),
                )
            )
    if positions != list(range(len(segment_rows))):
        raise RuntimeError("Segment positions are not contiguous and ordered")
    if len(segment_ids) != len(set(segment_ids)):
        raise RuntimeError("Segment IDs are not unique")

    flat_provenance: list[tuple[int, str, int, str]] = []
    segment_id_set = set(segment_ids)
    for row_number, row in enumerate(provenance_rows, start=1):
        if row.get("schema_version") != EXPECTED_PROVENANCE_SCHEMA:
            raise RuntimeError(f"Token provenance schema mismatch at row {row_number}")
        segment_id = str(row.get("segment_id", ""))
        if segment_id not in segment_id_set:
            raise RuntimeError(f"Token provenance segment mismatch at row {row_number}")
        position = row.get("segment_position")
        slot_id = row.get("slot_id")
        if not isinstance(position, int) or not isinstance(slot_id, int):
            raise RuntimeError(f"Malformed token provenance position at row {row_number}")
        flat_provenance.append((position, segment_id, slot_id, str(row.get("text", ""))))
    if flat_provenance != nested_provenance:
        raise RuntimeError("Token provenance JSONL differs from nested segment tokens")
    if len(provenance_rows) != total_tokens:
        raise RuntimeError("Token provenance row count differs from total token count")

    if normalization.get("schema_version") != EXPECTED_NORMALIZATION_SCHEMA:
        raise RuntimeError("Normalization report schema mismatch")
    if normalization.get("lecture_id") != lecture_id:
        raise RuntimeError("Normalization report lecture mismatch")
    if normalization.get("passed") is not True:
        raise RuntimeError("Normalization report has passed=false")
    view_reports = normalization.get("views")
    if not isinstance(view_reports, dict) or set(view_reports) != set(REQUIRED_VIEWS):
        raise RuntimeError("Normalization report view set mismatch")
    for view in REQUIRED_VIEWS:
        report = view_reports[view]
        if not isinstance(report, dict):
            raise RuntimeError(f"Normalization view report is invalid: {view}")
        if report.get("passed") is not True or report.get("source_accounting_closed") is not True:
            raise RuntimeError(f"Normalization view failed: {view}")
        if int(report.get("normalized_row_count", 0)) <= 0:
            raise RuntimeError(f"Normalization view has no rows: {view}")
        if int(report.get("normalized_word_count", 0)) <= 0:
            raise RuntimeError(f"Normalization view has no words: {view}")
        if Path(str(report.get("output", ""))).name != (
            f"{lecture_id}_{view}_normalized.jsonl"
        ):
            raise RuntimeError(f"Normalization output identity mismatch: {view}")

    if quality.get("schema_version") != EXPECTED_QUALITY_SCHEMA:
        raise RuntimeError("Quality report schema mismatch")
    if quality.get("passed") is not True:
        raise RuntimeError("Quality report has passed=false")
    validate_thresholds(quality)
    gates = quality.get("gates")
    if not isinstance(gates, dict) or set(gates) != EXPECTED_GATES:
        raise RuntimeError("Quality gate set mismatch")
    failed_gates = sorted(key for key, value in gates.items() if value is not True)
    if failed_gates:
        raise RuntimeError(f"Quality gates failed: {failed_gates}")
    if Path(str(quality.get("segment_jsonl", ""))).name != segment_name:
        raise RuntimeError("Quality report segment_jsonl identity mismatch")
    if Path(str(quality.get("normalization_report", ""))).name != normalization_name:
        raise RuntimeError("Quality report normalization_report identity mismatch")

    full = quality.get("full_lecture")
    pilot = quality.get("pilot_window")
    if not isinstance(full, dict) or not isinstance(pilot, dict):
        raise RuntimeError("Quality report summary sections are missing")
    if full.get("segment_count") != len(segment_rows):
        raise RuntimeError("Quality full_lecture segment_count mismatch")
    if full.get("total_tokens") != total_tokens:
        raise RuntimeError("Quality full_lecture total_tokens mismatch")
    if pilot.get("segment_count") != min(30, len(segment_rows)):
        raise RuntimeError("Quality pilot_window segment_count mismatch")
    if manifest.get("full_lecture") != full:
        raise RuntimeError("Manifest full_lecture differs from quality report")
    if manifest.get("pilot_window") != pilot:
        raise RuntimeError("Manifest pilot_window differs from quality report")
    if manifest.get("gates") != gates:
        raise RuntimeError("Manifest gates differ from quality report")

    return {
        "repository_commit": repository_commit,
        "segment_count": len(segment_rows),
        "token_provenance_row_count": len(provenance_rows),
        "total_tokens": total_tokens,
        "quality_gates": gates,
        "quality_thresholds": quality["thresholds"],
        "schemas": {
            "package": EXPECTED_SCHEMA,
            "quality": EXPECTED_QUALITY_SCHEMA,
            "segment": EXPECTED_SEGMENT_SCHEMA,
            "export": EXPECTED_EXPORT_SCHEMA,
            "token_provenance": EXPECTED_PROVENANCE_SCHEMA,
            "normalization": EXPECTED_NORMALIZATION_SCHEMA,
        },
    }


def validate_external_checksum(archive_path: Path) -> str:
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    if not checksum_path.is_file() or checksum_path.stat().st_size == 0:
        raise RuntimeError(f"Missing external ZIP checksum: {checksum_path}")
    lines = [
        line.strip()
        for line in checksum_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1 or "  " not in lines[0]:
        raise RuntimeError(f"Malformed external ZIP checksum: {checksum_path}")
    expected, filename = lines[0].split("  ", 1)
    if filename != archive_path.name:
        raise RuntimeError("External ZIP checksum filename mismatch")
    actual = sha256(archive_path)
    if expected != actual:
        raise RuntimeError("External ZIP checksum mismatch")
    return actual


def validate_archive(
    archive_path: Path,
    lecture_id: str,
    *,
    expected_repository_commit: str | None = None,
) -> dict[str, Any]:
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty repaired archive: {archive_path}")
    archive_sha = validate_external_checksum(archive_path)

    with zipfile.ZipFile(archive_path, "r") as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Corrupt ZIP member: {corrupt}")
        names = archive.namelist()
        file_names = [name for name in names if not name.endswith("/")]
        if len(file_names) != 8:
            raise RuntimeError(f"Repaired archive must contain exactly eight files; got {len(file_names)}")
        roots = {name.split("/", 1)[0] for name in file_names if "/" in name}
        if len(roots) != 1 or any("/" not in name for name in file_names):
            raise RuntimeError("Repaired archive must contain exactly one top-level package directory")
        package_root = next(iter(roots))
        relative_names = {name.split("/", 1)[1] for name in file_names}
        expected_names = package_filenames(lecture_id)
        if relative_names != expected_names:
            raise RuntimeError(
                "Repaired archive member contract mismatch. "
                f"expected={sorted(expected_names)} observed={sorted(relative_names)}"
            )
        member_by_relative = {
            name.split("/", 1)[1]: name
            for name in file_names
        }
        payloads = {
            relative: archive.read(member)
            for relative, member in member_by_relative.items()
        }
        manifest = json_from_bytes(payloads["PACKAGE_MANIFEST.json"], "PACKAGE_MANIFEST.json")
        checksums = parse_checksum_payload(payloads["SHA256SUMS.txt"])
        expected_checksum_names = set(source_filenames(lecture_id)) | {"PACKAGE_MANIFEST.json"}
        if set(checksums) != expected_checksum_names:
            raise RuntimeError(
                "SHA256SUMS entry contract mismatch. "
                f"expected={sorted(expected_checksum_names)} observed={sorted(checksums)}"
            )
        for filename in expected_checksum_names:
            if checksums[filename] != sha256_bytes(payloads[filename]):
                raise RuntimeError(f"SHA256SUMS mismatch for {filename}")

        if archive_path.name != f"{package_root}.zip":
            raise RuntimeError(
                f"Archive filename mismatch: {archive_path.name!r} != {package_root + '.zip'!r}"
            )
        payload_summary = validate_payload_contract(
            lecture_id=lecture_id,
            package_root=package_root,
            manifest=manifest,
            payloads={name: payloads[name] for name in source_filenames(lecture_id)},
            expected_repository_commit=expected_repository_commit,
        )

    return {
        "archive": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha,
        "archive_checksum_file": str(
            archive_path.with_suffix(archive_path.suffix + ".sha256")
        ),
        "member_count": len(file_names),
        "package_root": package_root,
        "manifest": manifest,
        **payload_summary,
    }


def validate_source_files(
    silver_root: Path,
    lecture_id: str,
    expected_repository_commit: str,
) -> dict[str, Any]:
    reconciled = silver_root / "reconciled_fixed"
    prefix = f"{lecture_id}_silver_v3_fixed"
    paths = {
        f"{prefix}_segment_level.jsonl": reconciled / f"{prefix}_segment_level.jsonl",
        f"{prefix}_token_provenance.jsonl": reconciled / f"{prefix}_token_provenance.jsonl",
        f"{prefix}.json": reconciled / f"{prefix}.json",
        f"{prefix}.docx": reconciled / f"{prefix}.docx",
        f"{prefix}_quality_report.json": reconciled / f"{prefix}_quality_report.json",
        "silver_v3_normalization_report.json": silver_root / "silver_v3_normalization_report.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing required repaired outputs:\n" + "\n".join(missing))
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    quality = json_from_bytes(payloads[f"{prefix}_quality_report.json"], f"{prefix}_quality_report.json")
    manifest = {
        "schema_version": EXPECTED_SCHEMA,
        "lecture_id": lecture_id,
        "output_prefix": prefix,
        "repository_commit": expected_repository_commit,
        "quality_validation_schema": quality.get("schema_version"),
        "quality_passed": quality.get("passed"),
        "full_lecture": quality.get("full_lecture"),
        "pilot_window": quality.get("pilot_window"),
        "gates": quality.get("gates"),
        "files": [
            {
                "filename": name,
                "size_bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "source_path": str(paths[name]),
            }
            for name, payload in payloads.items()
        ],
    }
    return validate_payload_contract(
        lecture_id=lecture_id,
        package_root=f"{lecture_id}_silver_v3_repaired_{expected_repository_commit[:8]}",
        manifest=manifest,
        payloads=payloads,
        expected_repository_commit=expected_repository_commit,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finalize one Silver v3 lecture to the immutable Lecture 001 repaired package contract."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--silver-root", type=Path, required=True)
    parser.add_argument("--lecture-id", required=True)
    parser.add_argument("--copy-to", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    silver_root = args.silver_root.resolve()
    lecture_id = args.lecture_id
    output_prefix = f"{lecture_id}_silver_v3_fixed"
    reconciled = silver_root / "reconciled_fixed"
    quality_report = reconciled / f"{output_prefix}_quality_report.json"
    segment_jsonl = reconciled / f"{output_prefix}_segment_level.jsonl"
    normalization_report = silver_root / "silver_v3_normalization_report.json"
    export_dir = silver_root / "repaired_package"
    copy_to = args.copy_to or (silver_root / "final_package")
    commit = git_head(repo_root)

    required = [
        segment_jsonl,
        reconciled / f"{output_prefix}_token_provenance.jsonl",
        reconciled / f"{output_prefix}.json",
        reconciled / f"{output_prefix}.docx",
        normalization_report,
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing required Silver v3 outputs:\n" + "\n".join(missing))

    # Run the committed validator unchanged and with its immutable defaults.
    quality_command = [
        sys.executable,
        str(repo_root / "pipeline/silver_v3/validate_quality.py"),
        "--segment-jsonl",
        str(segment_jsonl),
        "--normalization-report",
        str(normalization_report),
        "--output",
        str(quality_report),
    ]
    quality_result = subprocess.run(
        quality_command, cwd=repo_root, text=True, capture_output=True, check=False
    )
    quality = read_json(quality_report) if quality_report.is_file() else {}
    if quality_result.returncode != 0 or quality.get("passed") is not True:
        raise RuntimeError(
            "Silver v3 output failed the immutable Lecture 001 quality contract.\n"
            f"gates={json.dumps(quality.get('gates', {}), sort_keys=True)}\n"
            f"stdout:\n{quality_result.stdout[-4000:]}\n"
            f"stderr:\n{quality_result.stderr[-4000:]}"
        )

    source_validation = validate_source_files(silver_root, lecture_id, commit)

    package_summary = run(
        [
            sys.executable,
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
            str(copy_to),
        ],
        repo_root,
    )

    archive = Path(package_summary["archive"])
    validation = validate_archive(
        archive,
        lecture_id,
        expected_repository_commit=commit,
    )
    copied_archive = package_summary.get("copied_archive")
    copied_validation = None
    if copied_archive:
        copied_validation = validate_archive(
            Path(copied_archive),
            lecture_id,
            expected_repository_commit=commit,
        )
        if copied_validation["archive_sha256"] != validation["archive_sha256"]:
            raise RuntimeError("Copied repaired archive differs from source archive")

    result = {
        "passed": True,
        "lecture_id": lecture_id,
        "repository_commit": commit,
        "quality_report": str(quality_report),
        "source_validation": source_validation,
        "package": package_summary,
        "validation": validation,
        "copied_validation": copied_validation,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
