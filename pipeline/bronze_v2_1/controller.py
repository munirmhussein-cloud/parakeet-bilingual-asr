\
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from pipeline.common.json_io import (
    atomic_write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 12 deterministic Whisper "
            "Large-v3 segment runners."
        )
    )

    parser.add_argument(
        "--lecture-id",
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--runner-count",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--max-concurrent-runners",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--model",
        default="large-v3",
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    commands = []

    for runner_id in range(
        args.runner_count
    ):
        command = [
            sys.executable,
            "-m",
            "pipeline.bronze_v2_1.worker",
            "--lecture-id",
            args.lecture_id,
            "--manifest",
            str(args.manifest),
            "--output-dir",
            str(args.output_dir),
            "--runner-id",
            str(runner_id),
            "--runner-count",
            str(args.runner_count),
            "--model",
            args.model,
            "--device",
            args.device,
            "--beam-size",
            str(args.beam_size),
            "--temperature",
            str(args.temperature),
        ]

        if args.download_root:
            command.extend(
                [
                    "--download-root",
                    str(args.download_root),
                ]
            )

        if args.force:
            command.append("--force")

        commands.append(
            {
                "runner_id":
                runner_id,
                "command":
                command,
            }
        )

    started = time.perf_counter()

    pending = list(commands)
    active = {}
    results = []

    while pending or active:
        while (
            pending
            and len(active)
            < args.max_concurrent_runners
        ):
            spec = pending.pop(0)

            process = subprocess.Popen(
                spec["command"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            active[process] = spec

            print(
                "Started runner",
                spec["runner_id"],
            )

        finished = [
            process
            for process in active
            if process.poll() is not None
        ]

        if not finished:
            time.sleep(1.0)
            continue

        for process in finished:
            spec = active.pop(process)

            stdout, stderr = (
                process.communicate()
            )

            print("=" * 88)
            print(
                "RUNNER",
                spec["runner_id"],
                "RETURN CODE",
                process.returncode,
            )
            print("=" * 88)

            if stdout:
                print(stdout)

            if stderr:
                print(stderr)

            results.append(
                {
                    "runner_id":
                    spec["runner_id"],
                    "returncode":
                    process.returncode,
                }
            )

    failures = [
        result
        for result in results
        if result["returncode"] != 0
    ]

    report = {
        "schema_version":
        "bronze_v2_1_controller_report_v1",
        "lecture_id":
        args.lecture_id,
        "manifest":
        str(args.manifest),
        "output_dir":
        str(args.output_dir),
        "runner_count":
        args.runner_count,
        "max_concurrent_runners":
        args.max_concurrent_runners,
        "wall_clock_seconds":
        round(
            time.perf_counter()
            - started,
            3,
        ),
        "successful_runner_count":
        len(results) - len(failures),
        "failed_runner_count":
        len(failures),
        "failed_runner_ids":
        [
            item["runner_id"]
            for item in failures
        ],
        "validation": {
            "all_runners_completed":
            len(results)
            == args.runner_count,
            "all_runners_succeeded":
            not failures,
            "passed":
            (
                len(results)
                == args.runner_count
                and not failures
            ),
        },
    }

    atomic_write_json(
        args.output_dir
        / "_controller_report.json",
        report,
    )

    print(report)

    return (
        0
        if report["validation"]["passed"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
