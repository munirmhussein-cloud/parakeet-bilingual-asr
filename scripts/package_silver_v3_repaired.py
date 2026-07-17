from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_SUFFIXES = (
    "_segment_level.jsonl",
    "_token_provenance.jsonl",
    ".json",
    ".docx",
    "_quality_report.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package repaired Silver v3 outputs with checksums and provenance.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--silver-root", type=Path, required=True)
    parser.add_argument("--lecture-id", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--copy-to", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reconciled = args.silver_root / "reconciled_fixed"
    source_files = [
        reconciled / f"{args.output_prefix}_segment_level.jsonl",
        reconciled / f"{args.output_prefix}_token_provenance.jsonl",
        reconciled / f"{args.output_prefix}.json",
        reconciled / f"{args.output_prefix}.docx",
        reconciled / f"{args.output_prefix}_quality_report.json",
        args.silver_root / "silver_v3_normalization_report.json",
    ]
    missing = [str(path) for path in source_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required repaired outputs:\n" + "\n".join(missing))

    quality = json.loads(source_files[4].read_text(encoding="utf-8"))
    if not quality.get("passed"):
        raise RuntimeError("Quality report has passed=false; refusing to package.")

    commit = git_head(args.repo_root)
    package_name = f"{args.lecture_id}_silver_v3_repaired_{commit[:8]}"
    package_dir = args.export_dir / package_name
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    files_manifest = []
    for source in source_files:
        destination = package_dir / source.name
        shutil.copy2(source, destination)
        files_manifest.append({
            "filename": destination.name,
            "size_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "source_path": str(source),
        })

    metadata = {
        "schema_version": "silver_v3_repaired_export_package_v1",
        "lecture_id": args.lecture_id,
        "output_prefix": args.output_prefix,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": commit,
        "quality_validation_schema": quality.get("schema_version"),
        "quality_passed": quality.get("passed"),
        "full_lecture": quality.get("full_lecture"),
        "pilot_window": quality.get("pilot_window"),
        "gates": quality.get("gates"),
        "files": files_manifest,
    }
    metadata_path = package_dir / "PACKAGE_MANIFEST.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum_path = package_dir / "SHA256SUMS.txt"
    checksum_lines = [f"{item['sha256']}  {item['filename']}" for item in files_manifest]
    checksum_lines.append(f"{sha256(metadata_path)}  {metadata_path.name}")
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    archive_path = args.export_dir / f"{package_name}.zip"
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(package_dir.iterdir()):
            archive.write(path, arcname=f"{package_name}/{path.name}")

    archive_sha = sha256(archive_path)
    archive_checksum_path = args.export_dir / f"{archive_path.name}.sha256"
    archive_checksum_path.write_text(f"{archive_sha}  {archive_path.name}\n", encoding="utf-8")

    copied_archive = None
    copied_checksum = None
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        copied_archive = args.copy_to / archive_path.name
        copied_checksum = args.copy_to / archive_checksum_path.name
        shutil.copy2(archive_path, copied_archive)
        shutil.copy2(archive_checksum_path, copied_checksum)

    result = {
        "passed": True,
        "repository_commit": commit,
        "package_dir": str(package_dir),
        "archive": str(archive_path),
        "archive_sha256": archive_sha,
        "archive_checksum_file": str(archive_checksum_path),
        "copied_archive": str(copied_archive) if copied_archive else None,
        "copied_checksum_file": str(copied_checksum) if copied_checksum else None,
        "file_count": len(files_manifest) + 2,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
