#!/usr/bin/env python3

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


DEFAULT_REPO = Path(
    "/content/parakeet-bilingual-asr"
)

DEFAULT_DRIVE = Path(
    "/content/drive/MyDrive/parakeet-bilingual-asr"
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

    os.replace(temporary, path)


def select_runner(repo: Path) -> Path:
    candidates = [
        repo
        / "scripts"
        / "run_silver_v3_batch_adaptive.py",

        repo
        / "scripts"
        / "run_silver_v3_batch.py",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Neither Silver v3 batch runner exists:\n"
        + "\n".join(str(path) for path in candidates)
    )


def stream_command(
    command: list[str],
    cwd: Path,
    lecture_id: str,
) -> int:
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
        line = line.rstrip()

        if line:
            print(
                f"[{lecture_id}] {line}",
                flush=True,
            )

    return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Silver v3 one lecture at a time, "
            "delegating validation and resumability "
            "to the established Silver v3 batch runner."
        )
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO,
    )

    parser.add_argument(
        "--drive-root",
        type=Path,
        default=DEFAULT_DRIVE,
    )

    parser.add_argument(
        "--start",
        type=int,
        default=17,
    )

    parser.add_argument(
        "--end",
        type=int,
        default=104,
    )

    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
    )

    args = parser.parse_args()

    if not 1 <= args.start <= 104:
        raise ValueError("--start must be between 1 and 104")

    if not 1 <= args.end <= 104:
        raise ValueError("--end must be between 1 and 104")

    if args.start > args.end:
        raise ValueError("--start cannot exceed --end")

    repo = args.repo_root.resolve()
    drive = args.drive_root.resolve()
    runner = select_runner(repo)

    state_path = (
        drive
        / "production_a100"
        / "pipeline_state"
        / "silver_v3_sequential_summary.json"
    )

    lecture_ids = [
        f"lecture_{number:03d}"
        for number in range(
            args.start,
            args.end + 1,
        )
    ]

    state: dict[str, Any] = {
        "schema_version":
            "silver_v3_sequential_summary_v2",
        "runner":
            str(runner),
        "start":
            args.start,
        "end":
            args.end,
        "started_at":
            utc_now(),
        "updated_at":
            utc_now(),
        "finished_at":
            None,
        "results":
            {},
    }

    atomic_write_json(
        state_path,
        state,
    )

    print("=" * 96)
    print("SILVER v3 STREAMLINED SEQUENTIAL RUN")
    print("=" * 96)
    print("Underlying runner:", runner)
    print(
        "Lecture range:",
        f"{lecture_ids[0]}–{lecture_ids[-1]}",
    )
    print("State:", state_path)
    print()

    if args.plan_only:
        for index, lecture_id in enumerate(
            lecture_ids,
            start=1,
        ):
            print(
                f"{index:>3}/{len(lecture_ids)} "
                f"{lecture_id}"
            )

        return 0

    overall_started = time.monotonic()

    try:
        for index, lecture_id in enumerate(
            lecture_ids,
            start=1,
        ):
            print()
            print("=" * 96)
            print(
                f"{index}/{len(lecture_ids)} "
                f"STARTING {lecture_id}"
            )
            print("=" * 96)

            command = [
                sys.executable,
                "-u",
                str(runner),
                "--lectures",
                lecture_id,
            ]

            result = {
                "status": "running",
                "lecture_id": lecture_id,
                "command": command,
                "started_at": utc_now(),
            }

            state["results"][lecture_id] = result
            state["updated_at"] = utc_now()

            atomic_write_json(
                state_path,
                state,
            )

            lecture_started = time.monotonic()

            return_code = stream_command(
                command,
                repo,
                lecture_id,
            )

            elapsed = round(
                time.monotonic() - lecture_started,
                3,
            )

            status = (
                "completed"
                if return_code == 0
                else "failed"
            )

            result.update({
                "status": status,
                "return_code": return_code,
                "finished_at": utc_now(),
                "wall_seconds": elapsed,
            })

            state["results"][lecture_id] = result
            state["updated_at"] = utc_now()

            atomic_write_json(
                state_path,
                state,
            )

            if return_code == 0:
                print()
                print(
                    f"COMPLETED {lecture_id} "
                    f"in {elapsed:.1f}s"
                )
            else:
                print()
                print(
                    f"FAILED {lecture_id} "
                    f"after {elapsed:.1f}s "
                    f"with exit code {return_code}"
                )

                if not args.continue_after_failure:
                    print(
                        "Stopping at the first failure."
                    )
                    break

    except KeyboardInterrupt:
        state["interrupted"] = True
        state["updated_at"] = utc_now()
        state["finished_at"] = utc_now()

        atomic_write_json(
            state_path,
            state,
        )

        print()
        print(
            "Interrupted. Restart the same command "
            "to resume through the underlying runner's "
            "completion validation."
        )

        return 130

    statuses = [
        item.get("status")
        for item in state["results"].values()
    ]

    state["completed_count"] = statuses.count(
        "completed"
    )

    state["failure_count"] = statuses.count(
        "failed"
    )

    state["overall_wall_seconds"] = round(
        time.monotonic() - overall_started,
        3,
    )

    state["finished_at"] = utc_now()
    state["updated_at"] = utc_now()

    atomic_write_json(
        state_path,
        state,
    )

    print()
    print("=" * 96)
    print("SILVER v3 SEQUENTIAL RUN FINISHED")
    print("=" * 96)
    print(
        "Completed:",
        state["completed_count"],
    )
    print(
        "Failed:",
        state["failure_count"],
    )
    print("State:", state_path)

    return 0 if state["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
