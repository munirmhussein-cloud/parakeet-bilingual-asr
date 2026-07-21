
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".opus",
}

SEERAH_RE = re.compile(
    r"^seerah\D*0*(\d{1,3})(?!\d)",
    re.IGNORECASE,
)

LECTURE_RE = re.compile(
    r"^lecture[_\-\s]*0*(\d{1,3})(?!\d)",
    re.IGNORECASE,
)

TRANSIENT_TERMS = (
    "429",
    "too many requests",
    "throttl",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "service unavailable",
    "connection reset",
    "connection aborted",
    "502",
    "503",
    "504",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def parse_lecture_number(
    path: Path,
) -> int | None:

    for pattern in (
        SEERAH_RE,
        LECTURE_RE,
    ):
        match = pattern.match(path.stem)

        if match:
            return int(match.group(1))

    return None


def discover_audio(
    audio_root: Path,
) -> tuple[
    dict[str, Path],
    list[Path],
    dict[str, list[Path]],
]:

    grouped: dict[str, list[Path]] = {}
    unmatched: list[Path] = []

    for path in sorted(audio_root.iterdir()):
        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in SUPPORTED_AUDIO_EXTENSIONS
        ):
            continue

        number = parse_lecture_number(path)

        if number is None:
            unmatched.append(path)
            continue

        lecture_id = f"lecture_{number:03d}"

        grouped.setdefault(
            lecture_id,
            [],
        ).append(path)

    duplicates = {
        lecture_id: paths
        for lecture_id, paths in grouped.items()
        if len(paths) > 1
    }

    unique = {
        lecture_id: paths[0]
        for lecture_id, paths in grouped.items()
        if len(paths) == 1
    }

    return unique, unmatched, duplicates


def read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def validate_segment_jsonl(
    path: Path,
    lecture_id: str,
) -> int:

    if (
        not path.is_file()
        or path.stat().st_size == 0
    ):
        raise RuntimeError(
            f"Missing Silver v3 segment JSONL:\n{path}"
        )

    count = 0
    positions = []
    segment_ids: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL at "
                    f"{path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise RuntimeError(
                    f"Non-object JSONL row at "
                    f"{path}:{line_number}"
                )

            observed_lecture = row.get(
                "lecture_id"
            )

            if (
                observed_lecture is not None
                and observed_lecture != lecture_id
            ):
                raise RuntimeError(
                    f"Lecture mismatch at "
                    f"{path}:{line_number}: "
                    f"{observed_lecture}"
                )

            segment_id = row.get("segment_id")

            if segment_id:
                if segment_id in segment_ids:
                    raise RuntimeError(
                        f"Duplicate segment ID: {segment_id}"
                    )

                segment_ids.add(segment_id)

            position = row.get(
                "segment_position",
                row.get("segment_index"),
            )

            if isinstance(position, int):
                positions.append(position)

            count += 1

    if count == 0:
        raise RuntimeError(
            f"Silver v3 JSONL contains no rows:\n{path}"
        )

    if positions and positions != sorted(positions):
        raise RuntimeError(
            f"Silver v3 rows are not ordered:\n{path}"
        )

    return count


def validate_production_report(
    path: Path,
    lecture_id: str,
) -> dict[str, Any]:

    if (
        not path.is_file()
        or path.stat().st_size == 0
    ):
        raise RuntimeError(
            f"Missing Silver v3 production report:\n{path}"
        )

    report = read_json(path)

    if not isinstance(report, dict):
        raise RuntimeError(
            f"Production report is not an object:\n{path}"
        )

    if report.get("schema_version") not in {
        "silver_v3_production_run_report_v1",
        "silver_v3_production_run_report_v2",
    }:
        raise RuntimeError(
            "Unexpected Silver v3 report schema: "
            f"{report.get('schema_version')!r}"
        )

    if report.get("lecture_id") != lecture_id:
        raise RuntimeError(
            f"Lecture ID mismatch in report:\n{path}"
        )

    return report


def validate_normalization_report(
    path: Path,
) -> None:

    if (
        not path.is_file()
        or path.stat().st_size == 0
    ):
        raise RuntimeError(
            f"Missing normalization report:\n{path}"
        )

    payload = read_json(path)

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Normalization report is invalid:\n{path}"
        )


def validate_zip(
    zip_path: Path,
    lecture_id: str,
) -> int:

    if (
        not zip_path.is_file()
        or zip_path.stat().st_size == 0
    ):
        raise RuntimeError(
            f"Missing Silver v3 final ZIP:\n{zip_path}"
        )

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as archive:

        corrupt = archive.testzip()

        if corrupt is not None:
            raise RuntimeError(
                f"Corrupt ZIP member: {corrupt}"
            )

        members = archive.namelist()

    prefix = (
        f"{lecture_id}/silver_v3_final/"
    )

    if not any(
        member.startswith(prefix)
        for member in members
    ):
        raise RuntimeError(
            f"ZIP missing prefix: {prefix}"
        )

    return len(members)


def completion_is_valid(
    marker_path: Path,
    audio_path: Path,
    segment_path: Path,
    report_path: Path,
    zip_path: Path,
    lecture_id: str,
) -> tuple[bool, str]:

    try:
        # Adopt valid legacy output even if the marker is absent.
        segment_count = validate_segment_jsonl(
            segment_path,
            lecture_id,
        )

        validate_production_report(
            report_path,
            lecture_id,
        )

        if marker_path.is_file():
            marker = read_json(marker_path)

            if marker.get("status") != "completed":
                return False, "marker_not_completed"

            if marker.get("lecture_id") != lecture_id:
                return False, "marker_lecture_mismatch"

            if marker.get("source_audio") != str(audio_path):
                return False, "source_audio_changed"

            if (
                marker.get("source_audio_size_bytes")
                != audio_path.stat().st_size
            ):
                return False, "source_audio_size_changed"

        if zip_path.is_file():
            validate_zip(
                zip_path,
                lecture_id,
            )

        return (
            True,
            f"validated_existing_output:{segment_count}_rows",
        )

    except Exception as exc:
        return False, f"validation_failed:{exc}"


def create_final_zip(
    lecture_id: str,
    silver_root: Path,
    zip_path: Path,
) -> dict[str, Any]:

    reconciled = (
        silver_root
        / "reconciled_fixed"
    )

    source_files = sorted(
        path
        for path in reconciled.rglob("*")
        if path.is_file()
    )

    additional_files = [
        silver_root
        / "silver_v3_normalization_report.json",
        silver_root
        / "silver_v3_whole_hosted_report.json",
        silver_root
        / "silver_v3_canonical_20s_hosted_report.json",
        silver_root
        / "silver_v3_context_10s_stride_5s_hosted_report.json",
        silver_root
        / "silver_v3_local_2p5s_contiguous_hosted_report.json",
    ]

    for path in additional_files:
        if path.is_file():
            source_files.append(path)

    source_files = sorted(set(source_files))

    if not source_files:
        raise RuntimeError(
            f"No final Silver v3 files to package:\n"
            f"{silver_root}"
        )

    local_zip = (
        Path("/content")
        / f"{lecture_id}_silver_v3_final.zip"
    )

    local_zip.unlink(missing_ok=True)
    zip_path.unlink(missing_ok=True)

    with zipfile.ZipFile(
        local_zip,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:

        for source_file in source_files:
            if source_file.is_relative_to(reconciled):
                relative = source_file.relative_to(
                    reconciled
                )

                archive_path = (
                    Path(lecture_id)
                    / "silver_v3_final"
                    / "reconciled_fixed"
                    / relative
                )
            else:
                archive_path = (
                    Path(lecture_id)
                    / "silver_v3_final"
                    / "reports"
                    / source_file.name
                )

            archive.write(
                source_file,
                arcname=archive_path.as_posix(),
            )

    member_count = validate_zip(
        local_zip,
        lecture_id,
    )

    zip_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        local_zip,
        zip_path,
    )

    if (
        not zip_path.is_file()
        or zip_path.stat().st_size
        != local_zip.stat().st_size
    ):
        raise RuntimeError(
            f"Drive ZIP copy validation failed:\n"
            f"{zip_path}"
        )

    local_zip.unlink(missing_ok=True)

    return {
        "zip_path": str(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_member_count": member_count,
    }


def should_retry(text: str) -> bool:
    normalized = text.lower()

    return any(
        term in normalized
        for term in TRANSIENT_TERMS
    )


def run_process(
    command: list[str],
    repo_root: Path,
    log_path: Path,
    append: bool,
) -> tuple[int, str]:

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mode = "a" if append else "w"

    with log_path.open(
        mode,
        encoding="utf-8",
    ) as log:

        log.write(
            "\n$ "
            + " ".join(command)
            + "\n\n"
        )

        log.flush()

        process = subprocess.run(
            command,
            cwd=str(repo_root),
            env=os.environ.copy(),
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    tail = "\n".join(
        log_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()[-160:]
    )

    return process.returncode, tail


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Silver v3 over discovered lectures "
            "with bounded lecture concurrency and "
            "resumable completion markers."
        )
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(
            "/content/parakeet-bilingual-asr"
        ),
    )

    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/"
            "parakeet-bilingual-asr"
        ),
    )

    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/"
            "parakeet-bilingual-asr/audio"
        ),
    )

    parser.add_argument(
        "--lecture-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--view-workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--lectures",
        nargs="*",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.lecture_workers < 1:
        raise ValueError(
            "--lecture-workers must be at least 1"
        )

    if args.view_workers < 1:
        raise ValueError(
            "--view-workers must be at least 1"
        )

    nvidia_key = os.environ.get(
        "NVIDIA_API_KEY",
        "",
    ).strip()

    if not nvidia_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not configured."
        )

    runner = (
        args.repo_root
        / "scripts"
        / "run_silver_v3_production_lecture_tolerant.py"
    )

    production_root = (
        args.drive_root
        / "production_a100"
        / "lectures"
    )

    log_root = (
        args.drive_root
        / "production_a100"
        / "pipeline_logs"
        / "silver_v3_batch"
    )

    summary_path = (
        args.drive_root
        / "production_a100"
        / "pipeline_state"
        / "silver_v3_batch_summary.json"
    )

    for required in (
        args.repo_root,
        args.audio_root,
        runner,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    audio_map, unmatched, duplicates = discover_audio(
        args.audio_root
    )

    if duplicates:
        details = []

        for lecture_id, paths in sorted(
            duplicates.items()
        ):
            details.append(
                lecture_id
                + ":\n"
                + "\n".join(
                    f"  - {path}"
                    for path in paths
                )
            )

        raise RuntimeError(
            "Duplicate lecture audio files:\n\n"
            + "\n\n".join(details)
        )

    if args.lectures:
        requested: set[str] = set()

        for value in args.lectures:
            match = re.search(
                r"(\d{1,3})",
                value,
            )

            if not match:
                raise ValueError(
                    f"Invalid lecture value: {value}"
                )

            requested.add(
                f"lecture_{int(match.group(1)):03d}"
            )

        missing = sorted(
            requested - set(audio_map)
        )

        if missing:
            raise FileNotFoundError(
                "Requested lectures not discovered:\n"
                + "\n".join(missing)
            )

        audio_map = {
            lecture_id: audio_map[lecture_id]
            for lecture_id in sorted(requested)
        }

    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(args.repo_root),
        text=True,
        capture_output=True,
        check=False,
    )

    commit = (
        git_result.stdout.strip()
        if git_result.returncode == 0
        else "unknown"
    )

    summary: dict[str, Any] = {
        "schema_version":
        "silver_v3_parallel_batch_summary_v1",
        "repository_commit": commit,
        "lecture_workers": args.lecture_workers,
        "view_workers": args.view_workers,
        "maximum_hosted_concurrency":
        args.lecture_workers * args.view_workers,
        "max_attempts": args.max_attempts,
        "lecture_count": len(audio_map),
        "unmatched_audio_files": [
            str(path)
            for path in unmatched
        ],
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "finished_at": None,
        "completed": False,
        "results": {},
    }

    summary_lock = threading.Lock()
    print_lock = threading.Lock()

    def save_result(
        lecture_id: str,
        result: dict[str, Any],
    ) -> None:

        with summary_lock:
            summary["results"][lecture_id] = result
            summary["updated_at"] = utc_now()

            atomic_write_json(
                summary_path,
                summary,
            )

    atomic_write_json(
        summary_path,
        summary,
    )

    print("=" * 92)
    print("SILVER v3 BATCH PLAN")
    print("=" * 92)

    print("Repository commit:", commit)
    print("Lectures:", len(audio_map))
    print("Lecture workers:", args.lecture_workers)
    print("View workers:", args.view_workers)
    print(
        "Maximum hosted concurrency:",
        args.lecture_workers * args.view_workers,
    )
    print("Summary:", summary_path)

    for index, (
        lecture_id,
        audio_path,
    ) in enumerate(
        sorted(audio_map.items()),
        start=1,
    ):
        print(
            f"{index:>3}. {lecture_id} | "
            f"{audio_path.name}"
        )

    if args.plan_only:
        summary["completed"] = True
        summary["finished_at"] = utc_now()
        summary["updated_at"] = utc_now()

        atomic_write_json(
            summary_path,
            summary,
        )

        return 0

    def run_one(
        lecture_id: str,
        audio_path: Path,
    ) -> dict[str, Any]:

        lecture_root = (
            production_root
            / lecture_id
        )

        silver_root = (
            lecture_root
            / "silver_v3"
        )

        reconciled = (
            silver_root
            / "reconciled_fixed"
        )

        output_prefix = (
            f"{lecture_id}_silver_v3_fixed"
        )

        segment_path = (
            reconciled
            / f"{output_prefix}_segment_level.jsonl"
        )

        report_path = (
            reconciled
            / f"{output_prefix}_production_run_report.json"
        )

        normalization_report = (
            silver_root
            / "silver_v3_normalization_report.json"
        )

        final_package_root = (
            silver_root
            / "final_package"
        )

        zip_path = (
            final_package_root
            / f"{lecture_id}_silver_v3_final.zip"
        )

        marker_path = (
            silver_root
            / ".silver_v3_complete.json"
        )

        log_path = (
            log_root
            / f"{lecture_id}.log"
        )

        source_record = (
            lecture_root
            / "pipeline_state"
            / "source_audio.txt"
        )

        silver_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        source_record.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        source_record.write_text(
            str(audio_path) + "\n",
            encoding="utf-8",
        )

        valid, reason = completion_is_valid(
            marker_path,
            audio_path,
            segment_path,
            report_path,
            zip_path,
            lecture_id,
        )

        if valid and not args.overwrite:
            # Create the lean ZIP if legacy outputs exist
            # but the final package has not yet been made.
            if not zip_path.is_file():
                zip_report = create_final_zip(
                    lecture_id,
                    silver_root,
                    zip_path,
                )
            else:
                zip_report = {
                    "zip_path": str(zip_path),
                    "zip_size_bytes":
                    zip_path.stat().st_size,
                    "zip_member_count":
                    validate_zip(zip_path, lecture_id),
                }

            segment_count = validate_segment_jsonl(
                segment_path,
                lecture_id,
            )

            adopted_marker = {
                "schema_version":
                "silver_v3_completion_marker_v1",
                "status": "completed",
                "reason": reason,
                "lecture_id": lecture_id,
                "repository_commit": commit,
                "source_audio": str(audio_path),
                "source_audio_size_bytes":
                audio_path.stat().st_size,
                "source_audio_sha256":
                sha256_file(audio_path),
                "silver_root": str(silver_root),
                "segment_jsonl": str(segment_path),
                "segment_count": segment_count,
                "production_report": str(report_path),
                **zip_report,
                "adopted_existing_output": True,
                "finished_at": utc_now(),
            }

            atomic_write_json(
                marker_path,
                adopted_marker,
            )

            result = {
                **adopted_marker,
                "status": "skipped",
            }

            save_result(
                lecture_id,
                result,
            )

            with print_lock:
                print(
                    f"SKIPPED {lecture_id} | "
                    f"rows={segment_count}"
                )

            return result

        if args.overwrite and silver_root.exists():
            shutil.rmtree(silver_root)

            silver_root.mkdir(
                parents=True,
                exist_ok=True,
            )

        started_at = utc_now()
        started = time.monotonic()

        running = {
            "status": "running",
            "lecture_id": lecture_id,
            "source_audio": str(audio_path),
            "silver_root": str(silver_root),
            "segment_jsonl": str(segment_path),
            "log": str(log_path),
            "started_at": started_at,
        }

        save_result(
            lecture_id,
            running,
        )

        with print_lock:
            print(
                f"STARTED {lecture_id} | "
                f"{audio_path.name}"
            )

        command = [
            sys.executable,
            str(runner),
            "--repo-root",
            str(args.repo_root),
            "--lecture-id",
            lecture_id,
            "--audio",
            str(audio_path),
            "--silver-root",
            str(silver_root),
            "--view-workers",
            str(args.view_workers),
        ]

        try:
            last_error = None

            for attempt in range(
                1,
                args.max_attempts + 1,
            ):
                return_code, tail = run_process(
                    command,
                    args.repo_root,
                    log_path,
                    append=attempt > 1,
                )

                if return_code == 0:
                    break

                last_error = (
                    f"Attempt {attempt} returned "
                    f"{return_code}\n{tail}"
                )

                if (
                    attempt >= args.max_attempts
                    or not should_retry(tail)
                ):
                    raise RuntimeError(last_error)

                delay = min(
                    180,
                    20 * (2 ** (attempt - 1)),
                )

                with print_lock:
                    print(
                        f"RETRY {lecture_id} | "
                        f"attempt={attempt + 1} | "
                        f"delay={delay}s"
                    )

                time.sleep(delay)

            segment_count = validate_segment_jsonl(
                segment_path,
                lecture_id,
            )

            validate_production_report(
                report_path,
                lecture_id,
            )

            validate_normalization_report(
                normalization_report,
            )

            zip_report = create_final_zip(
                lecture_id,
                silver_root,
                zip_path,
            )

            result = {
                "schema_version":
                "silver_v3_completion_marker_v1",
                "status": "completed",
                "lecture_id": lecture_id,
                "repository_commit": commit,
                "source_audio": str(audio_path),
                "source_audio_size_bytes":
                audio_path.stat().st_size,
                "source_audio_sha256":
                sha256_file(audio_path),
                "silver_root": str(silver_root),
                "segment_jsonl": str(segment_path),
                "segment_size_bytes":
                segment_path.stat().st_size,
                "segment_count": segment_count,
                "production_report": str(report_path),
                "normalization_report":
                str(normalization_report),
                "log": str(log_path),
                "started_at": started_at,
                "finished_at": utc_now(),
                "wall_seconds": round(
                    time.monotonic() - started,
                    3,
                ),
                **zip_report,
            }

            atomic_write_json(
                marker_path,
                result,
            )

            save_result(
                lecture_id,
                result,
            )

            with print_lock:
                print(
                    f"COMPLETED {lecture_id} | "
                    f"rows={segment_count} | "
                    f"{result['wall_seconds']}s"
                )

            return result

        except Exception as exc:
            result = {
                "status": "failed",
                "lecture_id": lecture_id,
                "source_audio": str(audio_path),
                "silver_root": str(silver_root),
                "segment_jsonl": str(segment_path),
                "log": str(log_path),
                "started_at": started_at,
                "finished_at": utc_now(),
                "wall_seconds": round(
                    time.monotonic() - started,
                    3,
                ),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

            save_result(
                lecture_id,
                result,
            )

            with print_lock:
                print(
                    f"FAILED {lecture_id}: {exc}"
                )

            return result

    batch_started = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.lecture_workers
    ) as executor:

        futures = {
            executor.submit(
                run_one,
                lecture_id,
                audio_path,
            ): lecture_id

            for lecture_id, audio_path
            in sorted(audio_map.items())
        }

        for future in concurrent.futures.as_completed(
            futures
        ):
            lecture_id = futures[future]

            try:
                future.result()
            except Exception as exc:
                with print_lock:
                    print(
                        f"UNHANDLED {lecture_id}: {exc}"
                    )

    statuses = [
        result.get("status")
        for result in summary["results"].values()
    ]

    summary["completed_count"] = statuses.count(
        "completed"
    )

    summary["skipped_count"] = statuses.count(
        "skipped"
    )

    summary["failure_count"] = statuses.count(
        "failed"
    )

    summary["batch_wall_seconds"] = round(
        time.monotonic() - batch_started,
        3,
    )

    summary["finished_at"] = utc_now()
    summary["updated_at"] = utc_now()

    summary["completed"] = (
        summary["failure_count"] == 0
    )

    atomic_write_json(
        summary_path,
        summary,
    )

    print("\n" + "=" * 92)
    print("SILVER v3 BATCH FINISHED")
    print("=" * 92)

    print(
        "Completed:",
        summary["completed_count"],
    )

    print(
        "Skipped:",
        summary["skipped_count"],
    )

    print(
        "Failed:",
        summary["failure_count"],
    )

    print("Summary:", summary_path)

    return 0 if summary["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
