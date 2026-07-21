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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def stream(command: list[str], cwd: Path, prefix: str) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Silver v3 strictly one lecture at a time and accept completion only after "
            "the immutable Lecture 001 repair, quality, and export-package contract passes."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path("/content/parakeet-bilingual-asr"))
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/parakeet-bilingual-asr"),
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=104)
    parser.add_argument("--view-workers", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.start <= args.end <= 104:
        raise ValueError("Lecture range must satisfy 1 <= start <= end <= 104")

    repo_root = args.repo_root.resolve()
    drive_root = args.drive_root.resolve()
    batch_runner = repo_root / "scripts/run_silver_v3_batch.py"
    repair_finalizer = repo_root / "pipeline/silver_v3/finalize_fixed.py"
    contract_finalizer = repo_root / "scripts/finalize_silver_v3_contract.py"
    for required in (batch_runner, repair_finalizer, contract_finalizer):
        if not required.is_file():
            raise FileNotFoundError(required)

    state_path = (
        drive_root
        / "production_a100"
        / "pipeline_state"
        / "silver_v3_sequential_contract_summary.json"
    )
    lecture_ids = [f"lecture_{number:03d}" for number in range(args.start, args.end + 1)]
    state: dict[str, Any] = {
        "schema_version": "silver_v3_sequential_contract_summary_v2",
        "contract_reference": "lecture_001_silver_v3_repaired_export_package_v1",
        "repair_stage": str(repair_finalizer),
        "contract_stage": str(contract_finalizer),
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "finished_at": None,
        "results": {},
    }
    atomic_write_json(state_path, state)

    print("=" * 96)
    print("SILVER v3 SEQUENTIAL CONTRACT RUN")
    print("=" * 96)
    print("Lectures:", f"{lecture_ids[0]}–{lecture_ids[-1]}")
    print("View workers:", args.view_workers)
    print("Repair stage:", repair_finalizer)
    print("Contract stage:", contract_finalizer)
    print("State:", state_path)

    if args.plan_only:
        for index, lecture_id in enumerate(lecture_ids, 1):
            print(f"{index:>3}/{len(lecture_ids)} {lecture_id}")
        return 0

    try:
        for index, lecture_id in enumerate(lecture_ids, 1):
            print("\n" + "=" * 96)
            print(f"[{index}/{len(lecture_ids)}] {lecture_id}")
            print("=" * 96)
            started = time.monotonic()
            result: dict[str, Any] = {
                "status": "running",
                "started_at": utc_now(),
                "stages": {},
            }
            state["results"][lecture_id] = result
            state["updated_at"] = utc_now()
            atomic_write_json(state_path, state)

            batch_command = [
                sys.executable,
                "-u",
                str(batch_runner),
                "--lecture-workers", "1",
                "--view-workers", str(args.view_workers),
                "--max-attempts", str(args.max_attempts),
                "--lectures", lecture_id,
            ]
            if args.overwrite:
                batch_command.append("--overwrite")

            batch_code = stream(batch_command, repo_root, lecture_id)
            result["stages"]["hosted_and_base_construction"] = {"return_code": batch_code}
            state["updated_at"] = utc_now()
            atomic_write_json(state_path, state)
            if batch_code != 0:
                raise RuntimeError(f"Underlying Silver v3 batch runner failed with {batch_code}")

            silver_root = (
                drive_root
                / "production_a100"
                / "lectures"
                / lecture_id
                / "silver_v3"
            )

            repair_command = [
                sys.executable,
                "-u",
                str(repair_finalizer),
                "--repo-root", str(repo_root),
                "--silver-root", str(silver_root),
                "--lecture-id", lecture_id,
            ]
            repair_code = stream(repair_command, repo_root, lecture_id)
            result["stages"]["lecture_001_repair_finalization"] = {"return_code": repair_code}
            state["updated_at"] = utc_now()
            atomic_write_json(state_path, state)
            if repair_code != 0:
                marker = silver_root / ".silver_v3_complete.json"
                marker.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Lecture {lecture_id} failed the Lecture 001 repair/finalization stage"
                )

            finalize_command = [
                sys.executable,
                "-u",
                str(contract_finalizer),
                "--repo-root", str(repo_root),
                "--silver-root", str(silver_root),
                "--lecture-id", lecture_id,
                "--copy-to", str(silver_root / "final_package"),
            ]
            finalize_code = stream(finalize_command, repo_root, lecture_id)
            result["stages"]["quality_and_export_contract"] = {"return_code": finalize_code}
            state["updated_at"] = utc_now()
            atomic_write_json(state_path, state)
            if finalize_code != 0:
                marker = silver_root / ".silver_v3_complete.json"
                marker.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Lecture {lecture_id} failed the Lecture 001 repaired package contract"
                )

            elapsed = round(time.monotonic() - started, 3)
            result.update({
                "status": "completed",
                "finished_at": utc_now(),
                "wall_seconds": elapsed,
            })
            state["results"][lecture_id] = result
            state["updated_at"] = utc_now()
            atomic_write_json(state_path, state)
            print(f"COMPLETED {lecture_id} in {elapsed:.1f}s", flush=True)

    except KeyboardInterrupt:
        state["interrupted"] = True
        state["updated_at"] = utc_now()
        state["finished_at"] = utc_now()
        atomic_write_json(state_path, state)
        print("Interrupted; rerun the same command to resume.")
        return 130
    except Exception as error:
        lecture_id = lecture_id if "lecture_id" in locals() else "unknown"
        existing = state["results"].get(lecture_id, {})
        existing.update({
            "status": "failed",
            "finished_at": utc_now(),
            "error": str(error),
        })
        state["results"][lecture_id] = existing
        state["updated_at"] = utc_now()
        state["finished_at"] = utc_now()
        atomic_write_json(state_path, state)
        print(f"FAILED {lecture_id}: {error}", flush=True)
        return 1

    state["finished_at"] = utc_now()
    state["updated_at"] = utc_now()
    state["completed"] = True
    atomic_write_json(state_path, state)
    print("\nAll requested lectures satisfy the Lecture 001 Silver v3 contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
