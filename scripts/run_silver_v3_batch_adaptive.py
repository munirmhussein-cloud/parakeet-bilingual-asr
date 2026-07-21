
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VIEWS = (
    "whole",
    "canonical_20s",
    "context_10s_stride_5s",
    "local_2p5s_contiguous",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

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


def normalize_lecture_id(value: str) -> str:
    digits = "".join(
        character
        for character in value
        if character.isdigit()
    )

    if not digits:
        raise ValueError(
            f"Invalid lecture identifier: {value}"
        )

    return f"lecture_{int(digits):03d}"


def marker_valid(
    lectures_root: Path,
    lecture_id: str,
) -> bool:
    silver = (
        lectures_root
        / lecture_id
        / "silver_v3"
    )

    marker_path = (
        silver
        / ".silver_v3_complete.json"
    )

    marker = read_json(marker_path)

    if not marker:
        return False

    segment_path_value = marker.get(
        "segment_jsonl"
    )

    zip_path_value = (
        marker.get("zip_path")
        or marker.get("zip")
    )

    if not segment_path_value:
        return False

    segment_path = Path(
        str(segment_path_value)
    )

    if zip_path_value:
        zip_path = Path(
            str(zip_path_value)
        )
    else:
        zip_path = (
            silver
            / "final_package"
            / f"{lecture_id}_silver_v3_final.zip"
        )

    return bool(
        marker.get("status") == "completed"
        and segment_path.is_file()
        and segment_path.stat().st_size > 0
        and zip_path.is_file()
        and zip_path.stat().st_size > 0
    )


def compact_hosted_reports(
    lectures_root: Path,
    lecture_id: str,
) -> dict[str, int]:
    silver = (
        lectures_root
        / lecture_id
        / "silver_v3"
    )

    removed_characters = 0
    compacted_reports = 0

    for view in VIEWS:
        path = (
            silver
            / f"silver_v3_{view}_hosted_report.json"
        )

        report = read_json(path)

        if not report:
            continue

        changed = False

        for field in ("stdout", "stderr"):
            value = report.get(field)

            if isinstance(value, str) and value:
                removed_characters += len(value)

                report[field] = (
                    f"[removed by adaptive runner; "
                    f"{len(value)} characters]"
                )

                changed = True

        if changed:
            atomic_json(path, report)
            compacted_reports += 1

    return {
        "compacted_reports":
            compacted_reports,
        "removed_characters":
            removed_characters,
    }


def report_metrics(
    lectures_root: Path,
    lecture_id: str,
) -> dict[str, Any]:
    silver = (
        lectures_root
        / lecture_id
        / "silver_v3"
    )

    views: dict[str, Any] = {}

    for view in VIEWS:
        report = read_json(
            silver
            / f"silver_v3_{view}_hosted_report.json"
        ) or {}

        views[view] = {
            "passed":
                report.get("passed"),
            "workers":
                report.get("workers"),
            "wall_seconds":
                report.get("wall_seconds"),
            "expected_count":
                report.get("expected_count"),
            "valid_output_count":
                report.get("valid_output_count"),
            "missing_output_count":
                report.get("missing_output_count"),
            "invalid_output_count":
                report.get("invalid_output_count"),
            "return_code":
                report.get("return_code"),
        }

    return views


def tune_workers(
    *,
    current: int,
    metrics: dict[str, Any],
    minimum: int,
    maximum: int,
    lecture_wall_seconds: float | None = None,
) -> tuple[int, str]:

    local = metrics.get(
        "local_2p5s_contiguous",
        {},
    )

    passed = local.get("passed")
    wall = local.get("wall_seconds")
    expected = local.get("expected_count")
    valid = local.get("valid_output_count")
    missing = local.get("missing_output_count")
    invalid = local.get("invalid_output_count")
    return_code = local.get("return_code")

    failed = (
        passed is False
        or return_code not in (None, 0)
        or (missing or 0) > 0
        or (invalid or 0) > 0
    )

    if failed:
        reduced = max(
            minimum,
            current - 2,
        )

        return (
            reduced,
            "reduced_after_view_failure",
        )

    if (
        isinstance(lecture_wall_seconds, (int, float))
        and lecture_wall_seconds > 180
        and current > minimum
    ):
        reduced = max(
            minimum,
            current - 4,
        )

        return (
            reduced,
            "reduced_after_lecture_above_180s",
        )

    throughput = None

    if (
        isinstance(wall, (int, float))
        and wall > 0
        and isinstance(expected, int)
        and expected > 0
    ):
        throughput = expected / wall

    # Never scale beyond 12 without a dedicated benchmark.
    maximum = min(maximum, 12)

    # Eight workers is the known stable baseline.
    if current < 8 and not failed:
        return 8, "restored_known_stable_baseline"

    # Only test 10 or 12 when the current lecture completed
    # cleanly and throughput is clearly low without a long
    # total lecture wall time.
    if (
        current == 8
        and throughput is not None
        and throughput < 12
        and (
            lecture_wall_seconds is None
            or lecture_wall_seconds < 150
        )
        and maximum >= 10
    ):
        return 10, "controlled_trial_at_10_workers"

    if (
        current == 10
        and throughput is not None
        and throughput >= 14
        and (
            lecture_wall_seconds is None
            or lecture_wall_seconds < 120
        )
        and maximum >= 12
    ):
        return 12, "controlled_trial_at_12_workers"

    return current, "retained_stable_worker_count"


def tail_new_content(
    path: Path,
    offset: int,
) -> int:
    if not path.is_file():
        return offset

    size = path.stat().st_size

    if size < offset:
        offset = 0

    if size == offset:
        return offset

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        handle.seek(offset)
        text = handle.read()
        new_offset = handle.tell()

    lines = text.splitlines()

    # Avoid flooding Colab. Show stage commands,
    # summaries, errors, and every 100th progress item.
    for line in lines:
        show = False

        if line.startswith("$ "):
            show = True
        elif "Traceback" in line:
            show = True
        elif "FAILED" in line.upper():
            show = True
        elif '"passed":' in line:
            show = True
        elif '"wall_seconds":' in line:
            show = True
        elif '"failed":' in line:
            show = True
        elif '"completed":' in line:
            show = True
        else:
            match = None

            if line.startswith("[") and "/" in line:
                try:
                    prefix = line.split("]", 1)[0][1:]
                    completed, total = prefix.split("/", 1)

                    completed_number = int(completed)
                    total_number = int(total)

                    match = (
                        completed_number == 1
                        or completed_number == total_number
                        or completed_number % 100 == 0
                    )
                except Exception:
                    match = False

            show = bool(match)

        if show:
            print("    " + line, flush=True)

    return new_offset


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive resumable Silver v3 batch "
            "orchestrator for Colab."
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
        "--start-workers",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--minimum-workers",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--maximum-workers",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--start-lecture",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--end-lecture",
        type=int,
        default=104,
    )

    parser.add_argument(
        "--lectures",
        nargs="*",
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--keep-full-reports",
        action="store_true",
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
    )

    args = parser.parse_args()

    if not os.environ.get(
        "NVIDIA_API_KEY",
        "",
    ).strip():
        raise RuntimeError(
            "NVIDIA_API_KEY is not configured."
        )

    if not (
        1
        <= args.minimum_workers
        <= args.start_workers
        <= args.maximum_workers
    ):
        raise ValueError(
            "Worker limits must satisfy "
            "minimum <= start <= maximum."
        )

    existing_batch = (
        args.repo_root
        / "scripts"
        / "run_silver_v3_batch.py"
    )

    if not existing_batch.is_file():
        raise FileNotFoundError(
            existing_batch
        )

    lectures_root = (
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

    state_path = (
        args.drive_root
        / "production_a100"
        / "pipeline_state"
        / "silver_v3_adaptive_summary.json"
    )

    if args.lectures:
        lecture_ids = sorted({
            normalize_lecture_id(value)
            for value in args.lectures
        })
    else:
        lecture_ids = [
            f"lecture_{number:03d}"
            for number in range(
                args.start_lecture,
                args.end_lecture + 1,
            )
        ]

    workers = args.start_workers

    state: dict[str, Any] = {
        "schema_version":
            "silver_v3_adaptive_summary_v1",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "finished_at": None,
        "completed": False,
        "worker_limits": {
            "minimum":
                args.minimum_workers,
            "start":
                args.start_workers,
            "maximum":
                args.maximum_workers,
        },
        "results": {},
    }

    atomic_json(state_path, state)

    print("=" * 92, flush=True)
    print(
        "SILVER v3 ADAPTIVE BATCH",
        flush=True,
    )
    print("=" * 92, flush=True)
    print(
        "Lectures:",
        len(lecture_ids),
        flush=True,
    )
    print(
        "Starting hosted workers:",
        workers,
        flush=True,
    )
    print(
        "Worker range:",
        f"{args.minimum_workers}–"
        f"{args.maximum_workers}",
        flush=True,
    )
    print(
        "State:",
        state_path,
        flush=True,
    )

    if args.plan_only:
        # Plan quickly from existing summaries first. Avoid
        # hundreds of high-latency Drive stat/read operations.
        completed_from_summary = set()

        summary_candidates = (
            args.drive_root
            / "production_a100"
            / "pipeline_state"
        )

        for summary_name in (
            "silver_v3_batch_summary.json",
            "silver_v3_adaptive_summary.json",
        ):
            candidate = read_json(
                summary_candidates / summary_name
            ) or {}

            for lecture_id, result in (
                candidate.get("results", {}) or {}
            ).items():
                if result.get("status") in {
                    "completed",
                    "skipped",
                }:
                    completed_from_summary.add(
                        lecture_id
                    )

        skip_count = 0
        run_count = 0

        for lecture_id in lecture_ids:
            marker_path = (
                lectures_root
                / lecture_id
                / "silver_v3"
                / ".silver_v3_complete.json"
            )

            lightweight_marker_complete = False

            if marker_path.is_file():
                marker = read_json(marker_path) or {}

                lightweight_marker_complete = (
                    marker.get("status") == "completed"
                )

            if lecture_id in completed_from_summary:
                status = "skip"
                reason = "summary"
                skip_count += 1
            elif lightweight_marker_complete:
                status = "skip"
                reason = "completion_marker"
                skip_count += 1
            else:
                status = "run"
                reason = "not_recorded_complete"
                run_count += 1

            print(
                lecture_id,
                status,
                reason,
                flush=True,
            )

        print(
            f"Plan totals: skip={skip_count}, "
            f"run={run_count}",
            flush=True,
        )

        return 0

    batch_started = time.perf_counter()

    for position, lecture_id in enumerate(
        lecture_ids,
        start=1,
    ):
        if marker_valid(
            lectures_root,
            lecture_id,
        ):
            print(
                f"[{position}/{len(lecture_ids)}] "
                f"{lecture_id}: SKIPPED",
                flush=True,
            )

            state["results"][lecture_id] = {
                "status": "skipped",
                "reason":
                    "validated_completion_marker",
                "finished_at": utc_now(),
            }

            state["updated_at"] = utc_now()
            atomic_json(state_path, state)
            continue

        print(
            f"\n[{position}/{len(lecture_ids)}] "
            f"{lecture_id}: START",
            flush=True,
        )

        print(
            f"  Hosted workers: {workers}",
            flush=True,
        )

        command = [
            sys.executable,
            "-u",
            str(existing_batch),
            "--lectures",
            lecture_id,
            "--lecture-workers",
            "1",
            "--view-workers",
            str(workers),
            "--max-attempts",
            str(args.max_attempts),
        ]

        print(
            "  $ " + " ".join(command),
            flush=True,
        )

        lecture_started = time.perf_counter()

        process = subprocess.Popen(
            command,
            cwd=str(args.repo_root),
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        log_path = (
            log_root
            / f"{lecture_id}.log"
        )

        log_offset = 0

        assert process.stdout is not None

        while process.poll() is None:
            # Read any batch-level status without blocking.
            line = process.stdout.readline()

            if line:
                stripped = line.rstrip()

                if stripped:
                    print(
                        "  " + stripped,
                        flush=True,
                    )

            log_offset = tail_new_content(
                log_path,
                log_offset,
            )

            time.sleep(args.poll_seconds)

        for line in process.stdout:
            stripped = line.rstrip()

            if stripped:
                print(
                    "  " + stripped,
                    flush=True,
                )

        log_offset = tail_new_content(
            log_path,
            log_offset,
        )

        return_code = process.wait()

        wall_seconds = round(
            time.perf_counter()
            - lecture_started,
            3,
        )

        completed = marker_valid(
            lectures_root,
            lecture_id,
        )

        metrics = report_metrics(
            lectures_root,
            lecture_id,
        )

        compact_result = {
            "compacted_reports": 0,
            "removed_characters": 0,
        }

        if (
            completed
            and not args.keep_full_reports
        ):
            compact_result = (
                compact_hosted_reports(
                    lectures_root,
                    lecture_id,
                )
            )

        status = (
            "completed"
            if completed
            else "failed"
        )

        result = {
            "status": status,
            "lecture_id": lecture_id,
            "return_code": return_code,
            "wall_seconds": wall_seconds,
            "workers": workers,
            "view_metrics": metrics,
            "report_compaction":
                compact_result,
            "finished_at": utc_now(),
        }

        state["results"][lecture_id] = result

        print(
            f"  {status.upper()}: "
            f"{wall_seconds}s",
            flush=True,
        )

        local = metrics.get(
            "local_2p5s_contiguous",
            {},
        )

        print(
            "  Local view:",
            {
                "wall_seconds":
                    local.get("wall_seconds"),
                "workers":
                    local.get("workers"),
                "passed":
                    local.get("passed"),
                "expected":
                    local.get("expected_count"),
                "valid":
                    local.get("valid_output_count"),
            },
            flush=True,
        )

        previous_workers = workers

        workers, tuning_reason = tune_workers(
            current=workers,
            metrics=metrics,
            minimum=args.minimum_workers,
            maximum=args.maximum_workers,
            lecture_wall_seconds=wall_seconds,
        )

        print(
            f"  Next worker setting: "
            f"{previous_workers} → {workers} "
            f"({tuning_reason})",
            flush=True,
        )

        result["next_workers"] = workers
        result["tuning_reason"] = (
            tuning_reason
        )

        state["updated_at"] = utc_now()
        atomic_json(state_path, state)

        if not completed:
            print(
                "  Batch stopped after failure. "
                "Rerun the same command to resume.",
                flush=True,
            )

            break

    statuses = [
        result.get("status")
        for result in state["results"].values()
    ]

    state["completed_count"] = (
        statuses.count("completed")
    )

    state["skipped_count"] = (
        statuses.count("skipped")
    )

    state["failure_count"] = (
        statuses.count("failed")
    )

    state["batch_wall_seconds"] = round(
        time.perf_counter()
        - batch_started,
        3,
    )

    state["finished_at"] = utc_now()
    state["updated_at"] = utc_now()
    state["completed"] = (
        state["failure_count"] == 0
        and len(state["results"])
        == len(lecture_ids)
    )

    atomic_json(state_path, state)

    print("\n" + "=" * 92, flush=True)
    print(
        "ADAPTIVE BATCH FINISHED",
        flush=True,
    )
    print("=" * 92, flush=True)
    print(
        "Completed:",
        state["completed_count"],
        flush=True,
    )
    print(
        "Skipped:",
        state["skipped_count"],
        flush=True,
    )
    print(
        "Failed:",
        state["failure_count"],
        flush=True,
    )
    print(
        "State:",
        state_path,
        flush=True,
    )

    return 0 if state["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
