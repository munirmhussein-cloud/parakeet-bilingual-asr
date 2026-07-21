from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "silver_v3_repaired_export_package_v1"
EXPECTED_QUALITY_SCHEMA = "silver_v3_quality_validation_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def run(command: list[str], cwd: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout[-8000:]}\n"
            f"stderr:\n{result.stderr[-8000:]}"
        )
    output = result.stdout.strip()
    return json.loads(output) if output else {}


def validate_archive(archive_path: Path, lecture_id: str) -> dict[str, Any]:
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty repaired archive: {archive_path}")

    with zipfile.ZipFile(archive_path, "r") as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Corrupt ZIP member: {corrupt}")
        members = archive.namelist()
        manifest_names = [name for name in members if name.endswith("/PACKAGE_MANIFEST.json")]
        checksum_names = [name for name in members if name.endswith("/SHA256SUMS.txt")]
        if len(manifest_names) != 1 or len(checksum_names) != 1:
            raise RuntimeError("Repaired archive must contain one manifest and one checksum file")
        manifest = json.loads(archive.read(manifest_names[0]))
        checksums = archive.read(checksum_names[0]).decode("utf-8").splitlines()

        if manifest.get("schema_version") != EXPECTED_SCHEMA:
            raise RuntimeError(f"Unexpected package schema: {manifest.get('schema_version')!r}")
        if manifest.get("quality_validation_schema") != EXPECTED_QUALITY_SCHEMA:
            raise RuntimeError(
                f"Unexpected quality schema: {manifest.get('quality_validation_schema')!r}"
            )
        if manifest.get("lecture_id") != lecture_id:
            raise RuntimeError("Package lecture ID mismatch")
        if manifest.get("quality_passed") is not True:
            raise RuntimeError("Package quality_passed is not true")

        expected_files = {
            f"{lecture_id}_silver_v3_fixed_segment_level.jsonl",
            f"{lecture_id}_silver_v3_fixed_token_provenance.jsonl",
            f"{lecture_id}_silver_v3_fixed.json",
            f"{lecture_id}_silver_v3_fixed.docx",
            f"{lecture_id}_silver_v3_fixed_quality_report.json",
            "silver_v3_normalization_report.json",
        }
        manifest_files = {item.get("filename") for item in manifest.get("files", [])}
        if manifest_files != expected_files:
            raise RuntimeError(
                f"Package file contract mismatch. expected={sorted(expected_files)} "
                f"observed={sorted(manifest_files)}"
            )

        checksum_map: dict[str, str] = {}
        for line in checksums:
            if not line.strip():
                continue
            digest, filename = line.split("  ", 1)
            checksum_map[filename] = digest

        package_prefix = manifest_names[0].rsplit("/", 1)[0]
        for item in manifest.get("files", []):
            filename = item["filename"]
            payload = archive.read(f"{package_prefix}/{filename}")
            digest = hashlib.sha256(payload).hexdigest()
            if digest != item.get("sha256") or digest != checksum_map.get(filename):
                raise RuntimeError(f"Checksum mismatch for {filename}")
            if len(payload) != item.get("size_bytes"):
                raise RuntimeError(f"Size mismatch for {filename}")

        manifest_payload = archive.read(manifest_names[0])
        manifest_digest = hashlib.sha256(manifest_payload).hexdigest()
        if checksum_map.get("PACKAGE_MANIFEST.json") != manifest_digest:
            raise RuntimeError("Manifest checksum mismatch")

    return {
        "archive": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "member_count": len(members),
        "manifest": manifest,
    }


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

    required = [
        segment_jsonl,
        reconciled / f"{output_prefix}_token_provenance.jsonl",
        reconciled / f"{output_prefix}.json",
        reconciled / f"{output_prefix}.docx",
        normalization_report,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required Silver v3 outputs:\n" + "\n".join(missing))

    quality_command = [
        sys.executable,
        str(repo_root / "pipeline/silver_v3/validate_quality.py"),
        "--segment-jsonl", str(segment_jsonl),
        "--normalization-report", str(normalization_report),
        "--output", str(quality_report),
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

    package_summary = run([
        sys.executable,
        str(repo_root / "scripts/package_silver_v3_repaired.py"),
        "--repo-root", str(repo_root),
        "--silver-root", str(silver_root),
        "--lecture-id", lecture_id,
        "--output-prefix", output_prefix,
        "--export-dir", str(export_dir),
        "--copy-to", str(copy_to),
    ], repo_root)

    archive = Path(package_summary["archive"])
    validation = validate_archive(archive, lecture_id)
    copied_archive = package_summary.get("copied_archive")
    if copied_archive:
        copied_validation = validate_archive(Path(copied_archive), lecture_id)
        if copied_validation["archive_sha256"] != validation["archive_sha256"]:
            raise RuntimeError("Copied repaired archive differs from source archive")

    result = {
        "passed": True,
        "lecture_id": lecture_id,
        "quality_report": str(quality_report),
        "package": package_summary,
        "validation": validation,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
