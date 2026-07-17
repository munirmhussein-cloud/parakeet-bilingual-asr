from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_RECONCILED_SUFFIXES = (
    "_segment_level.jsonl",
    "_token_provenance.jsonl",
    ".json",
    ".docx",
    "_report.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def copy_file(
    source: Path,
    destination: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(source, destination)

    return {
        "path":
        destination.as_posix(),

        "size_bytes":
        destination.stat().st_size,

        "sha256":
        sha256_file(destination),
    }


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def create_zip(
    source_dir: Path,
    zip_path: Path,
) -> None:
    zip_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(
            source_dir.rglob("*")
        ):
            if not path.is_file():
                continue

            archive.write(
                path,
                arcname=(
                    Path(source_dir.name)
                    / path.relative_to(source_dir)
                ).as_posix(),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a portable validated Silver v3 "
            "lecture package."
        )
    )

    parser.add_argument(
        "--lecture-id",
        required=True,
    )

    parser.add_argument(
        "--lecture-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    lecture_root = (
        args.lecture_root.resolve()
    )

    silver_root = (
        lecture_root
        / "silver_v3"
    )

    reconciled_root = (
        silver_root
        / "reconciled"
    )

    manifest_root = (
        silver_root
        / "manifests"
    )

    runtime_root = (
        lecture_root
        / "runtime"
    )

    logs_root = (
        lecture_root
        / "logs"
    )

    output_prefix = (
        f"{args.lecture_id}_silver_v3"
    )

    report_path = (
        reconciled_root
        / f"{output_prefix}_report.json"
    )

    if not report_path.exists():
        raise FileNotFoundError(
            report_path
        )

    report = read_json(report_path)
    validation = report.get(
        "validation",
        {},
    )

    if validation.get("passed") is not True:
        raise RuntimeError(
            "Silver v3 validation has not passed:\n"
            + json.dumps(
                validation,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )

    expected_outputs = {
        "segment_level":
        reconciled_root
        / f"{output_prefix}_segment_level.jsonl",

        "token_provenance":
        reconciled_root
        / f"{output_prefix}_token_provenance.jsonl",

        "json":
        reconciled_root
        / f"{output_prefix}.json",

        "docx":
        reconciled_root
        / f"{output_prefix}.docx",

        "report":
        report_path,
    }

    missing = [
        str(path)
        for path in expected_outputs.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing final Silver v3 outputs:\n"
            + "\n".join(missing)
        )

    package_name = (
        f"{args.lecture_id}_silver_v3_package"
    )

    package_root = (
        args.output_root
        / package_name
    )

    zip_path = (
        args.output_root
        / f"{package_name}.zip"
    )

    if package_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Package already exists: "
                f"{package_root}"
            )

        shutil.rmtree(package_root)

    if zip_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"ZIP already exists: {zip_path}"
            )

        zip_path.unlink()

    started_at = datetime.now(
        timezone.utc
    ).isoformat()

    started_perf = time.perf_counter()

    copied_files = []


    # Final deliverables.
    for name, source in expected_outputs.items():
        copied_files.append(
            copy_file(
                source,
                package_root
                / "outputs"
                / source.name,
            )
        )


    # Canonical and inference view manifests.
    manifest_names = (
        f"{args.lecture_id}_whole.jsonl",
        f"{args.lecture_id}_canonical_20s.jsonl",
        f"{args.lecture_id}_canonical_manifest.jsonl",
        (
            f"{args.lecture_id}"
            "_context_10s_stride_5s.jsonl"
        ),
        (
            f"{args.lecture_id}"
            "_local_2p5s_contiguous.jsonl"
        ),
    )

    for name in manifest_names:
        source = manifest_root / name

        if source.exists():
            copied_files.append(
                copy_file(
                    source,
                    package_root
                    / "manifests"
                    / source.name,
                )
            )


    # Runtime validation and benchmark records.
    runtime_patterns = (
        "silver_v3*.json",
        "silver_v3*.txt",
    )

    for pattern in runtime_patterns:
        for source in sorted(
            runtime_root.glob(pattern)
        ):
            if source.is_file():
                copied_files.append(
                    copy_file(
                        source,
                        package_root
                        / "runtime"
                        / source.name,
                    )
                )


    # Relevant logs only.
    for source in sorted(
        logs_root.glob("silver_v3*.log")
    ):
        if source.is_file():
            copied_files.append(
                copy_file(
                    source,
                    package_root
                    / "logs"
                    / source.name,
                )
            )


    # Include diagnostic evidence when available.
    for source in sorted(
        reconciled_root.glob(
            f"{args.lecture_id}_silver_v3"
            "*diagnostic.json"
        )
    ):
        copied_files.append(
            copy_file(
                source,
                package_root
                / "diagnostics"
                / source.name,
            )
        )


    package_relative_files = []

    for path in sorted(
        package_root.rglob("*")
    ):
        if not path.is_file():
            continue

        package_relative_files.append(
            {
                "path":
                path.relative_to(
                    package_root
                ).as_posix(),

                "size_bytes":
                path.stat().st_size,

                "sha256":
                sha256_file(path),
            }
        )


    package_manifest = {
        "schema_version":
        "silver_v3_export_package_v1",

        "lecture_id":
        args.lecture_id,

        "created_at_utc":
        started_at,

        "source_lecture_root":
        str(lecture_root),

        "silver_v3_schema_version":
        report.get("schema_version"),

        "segment_count":
        report.get("segment_count"),

        "total_tokens":
        report.get("total_tokens"),

        "tier_distribution":
        report.get("tier_distribution"),

        "single_witness_token_count":
        report.get(
            "single_witness_token_count"
        ),

        "validation":
        validation,

        "source_output_sha256":
        report.get("sha256"),

        "file_count":
        len(package_relative_files),

        "files":
        package_relative_files,

        "excluded_categories": [
            "source audio",
            "audio view WAV files",
            "raw hosted Parakeet responses",
            "normalized intermediate views",
            "credentials and environment secrets",
        ],
    }

    package_manifest_path = (
        package_root
        / "PACKAGE_MANIFEST.json"
    )

    write_json(
        package_manifest_path,
        package_manifest,
    )


    checksums_path = (
        package_root
        / "SHA256SUMS"
    )

    checksum_lines = [
        (
            f"{item['sha256']}  "
            f"{item['path']}"
        )
        for item in package_relative_files
    ]

    checksum_lines.append(
        (
            f"{sha256_file(package_manifest_path)}  "
            "PACKAGE_MANIFEST.json"
        )
    )

    checksums_path.write_text(
        "\n".join(checksum_lines)
        + "\n",
        encoding="utf-8",
    )


    readme_path = (
        package_root
        / "README.md"
    )

    readme_path.write_text(
        (
            f"# {args.lecture_id} Silver v3 Package\n\n"
            "This package contains the validated Silver v3 "
            "outputs and reproducibility metadata for one "
            "lecture.\n\n"
            "## Final validation\n\n"
            f"- Segments: {report.get('segment_count')}\n"
            f"- Tokens: {report.get('total_tokens')}\n"
            "- Chronology errors: "
            f"{validation.get('chronology_error_count')}\n"
            "- Unaccounted observations: "
            f"{validation.get('unaccounted_observation_count')}\n"
            "- Remaining duplicate six-grams: "
            f"{validation.get('immediate_duplicate_6gram_count')}\n"
            "- Overlap collapses: "
            f"{validation.get('duplicate_overlap_collapse_count')}\n"
            "- Empty trailing segments: "
            f"{validation.get('empty_segment_count')}\n"
            "- Validation passed: "
            f"{validation.get('passed')}\n\n"
            "See `PACKAGE_MANIFEST.json` for the complete "
            "file inventory and provenance metadata.\n"
        ),
        encoding="utf-8",
    )


    # Rebuild package inventory to include README,
    # manifest and checksum file.
    complete_inventory = []

    for path in sorted(
        package_root.rglob("*")
    ):
        if not path.is_file():
            continue

        complete_inventory.append(
            {
                "path":
                path.relative_to(
                    package_root
                ).as_posix(),

                "size_bytes":
                path.stat().st_size,

                "sha256":
                sha256_file(path),
            }
        )


    create_zip(
        package_root,
        zip_path,
    )


    zip_sha256 = sha256_file(zip_path)

    export_report = {
        "schema_version":
        "silver_v3_export_report_v1",

        "lecture_id":
        args.lecture_id,

        "package_root":
        str(package_root),

        "zip_path":
        str(zip_path),

        "zip_size_bytes":
        zip_path.stat().st_size,

        "zip_sha256":
        zip_sha256,

        "package_file_count":
        len(complete_inventory),

        "wall_seconds":
        round(
            time.perf_counter()
            - started_perf,
            3,
        ),

        "validation_passed":
        validation.get("passed") is True,

        "passed":
        True,
    }

    export_report_path = (
        args.output_root
        / f"{package_name}_export_report.json"
    )

    write_json(
        export_report_path,
        export_report,
    )


    # Validate ZIP integrity.
    with zipfile.ZipFile(
        zip_path,
        mode="r",
    ) as archive:
        bad_member = archive.testzip()

        if bad_member is not None:
            raise RuntimeError(
                f"ZIP integrity failure: "
                f"{bad_member}"
            )


    print(
        json.dumps(
            export_report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
