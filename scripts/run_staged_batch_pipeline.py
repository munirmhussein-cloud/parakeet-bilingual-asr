from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGES = ("bronze_v3", "silver_v3", "silver_plus_v4")
SCHEMA_VERSION = "staged_bronze_v3_silver_v3_silver_plus_v4_batch_v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render(value: str, context: dict[str, str]) -> str:
    return value.format(**context)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def auto_workers(stage_config: dict[str, Any], lecture_count: int) -> int:
    configured = stage_config.get("max_workers", "auto")
    if str(configured).lower() != "auto":
        return max(1, min(lecture_count, int(configured)))

    cpu_count = max(1, os.cpu_count() or 1)
    resource = str(stage_config.get("resource", "cpu")).lower()

    if resource == "gpu":
        # One process per visible GPU by default. A stage can explicitly
        # override max_workers when the underlying runner safely shares a GPU.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible and visible != "-1":
            gpu_count = len([value for value in visible.split(",") if value.strip()])
        else:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                gpu_count = len([line for line in result.stdout.splitlines() if line.strip()])
            except FileNotFoundError:
                gpu_count = 0
        return max(1, min(lecture_count, gpu_count or 1))

    if resource == "io":
        return max(1, min(lecture_count, cpu_count * 2))

    return max(1, min(lecture_count, cpu_count))


def validate_config(config: dict[str, Any]) -> None:
    if "common" not in config or "stages" not in config:
        raise ValueError("Config requires top-level common and stages objects")

    missing = [stage for stage in STAGES if stage not in config["stages"]]
    if missing:
        raise ValueError(f"Config is missing required stages: {missing}")

    for stage in STAGES:
        stage_config = config["stages"][stage]
        command = stage_config.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError(f"{stage}.command must be a non-empty JSON list")
        expected = stage_config.get("expected_outputs")
        if not isinstance(expected, list) or not expected:
            raise ValueError(
                f"{stage}.expected_outputs must be non-empty so stage barriers cannot pass silently"
            )


def run_one(
    stage: str,
    lecture_id: str,
    stage_config: dict[str, Any],
    common: dict[str, str],
    log_root: Path,
    resume: bool,
) -> dict[str, Any]:
    started_at = utc_now()
    context = {**common, "lecture_id": lecture_id, "stage": stage}
    expected = [Path(render(path, context)) for path in stage_config["expected_outputs"]]

    if resume and all(path.exists() for path in expected):
        return {
            "lecture_id": lecture_id,
            "stage": stage,
            "status": "skipped",
            "reason": "all_expected_outputs_exist",
            "expected_outputs": [str(path) for path in expected],
            "started_at": started_at,
            "finished_at": utc_now(),
        }

    command = [render(str(part), context) for part in stage_config["command"]]
    env = os.environ.copy()
    env.update(
        {
            key: render(str(value), context)
            for key, value in stage_config.get("env", {}).items()
        }
    )
    cwd = Path(render(stage_config.get("cwd", common["repo_root"]), context))
    log_dir = log_root / stage
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{lecture_id}.log"

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"{stage} failed for {lecture_id} with return code {result.returncode}; "
            f"log={log_path}"
        )

    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(
            f"{stage} completed for {lecture_id} but expected outputs are missing: {missing}; "
            f"log={log_path}"
        )

    return {
        "lecture_id": lecture_id,
        "stage": stage,
        "status": "completed",
        "log": str(log_path),
        "expected_outputs": [str(path) for path in expected],
        "started_at": started_at,
        "finished_at": utc_now(),
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Bronze v3 -> Silver v3 -> Silver+ v4 sequentially, with a hard "
            "barrier between stages and maximum safe lecture parallelism inside each stage."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lectures", nargs="+", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    validate_config(config)
    common = {key: str(value) for key, value in config["common"].items()}
    log_root = Path(common.get("log_root", Path(common["drive_root"]) / "pipeline_logs"))
    summary_path = args.summary or (log_root / "latest_batch_summary.json")
    results: list[dict[str, Any]] = []
    started_at = utc_now()

    plan = {
        stage: {
            "workers": auto_workers(config["stages"][stage], len(args.lectures)),
            "resource": config["stages"][stage].get("resource", "cpu"),
        }
        for stage in STAGES
    }
    print(json.dumps({"lectures": args.lectures, "stage_order": STAGES, "plan": plan}, indent=2))
    if args.plan_only:
        return 0

    try:
        for stage in STAGES:
            stage_config = config["stages"][stage]
            max_workers = plan[stage]["workers"]
            print(
                f"\n=== {stage.upper()} | lectures={len(args.lectures)} | "
                f"workers={max_workers} | resource={plan[stage]['resource']} ==="
            )
            stage_results: list[dict[str, Any]] = []

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures: dict[Future[dict[str, Any]], str] = {
                    pool.submit(
                        run_one,
                        stage,
                        lecture_id,
                        stage_config,
                        common,
                        log_root,
                        args.resume,
                    ): lecture_id
                    for lecture_id in args.lectures
                }

                try:
                    for future in as_completed(futures):
                        result = future.result()
                        stage_results.append(result)
                        print(json.dumps(result, sort_keys=True))
                except Exception:
                    for pending in futures:
                        pending.cancel()
                    raise

            results.extend(sorted(stage_results, key=lambda item: item["lecture_id"]))
            print(f"=== {stage.upper()} COMPLETE; BARRIER RELEASED ===")

    except Exception as exc:
        failed_summary = {
            "schema_version": SCHEMA_VERSION,
            "lectures": args.lectures,
            "stage_order": list(STAGES),
            "plan": plan,
            "results": results,
            "passed": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "started_at": started_at,
            "finished_at": utc_now(),
        }
        write_summary(summary_path, failed_summary)
        print(json.dumps({"completed": False, "summary": str(summary_path), "error": str(exc)}, indent=2))
        return 1

    summary = {
        "schema_version": SCHEMA_VERSION,
        "lectures": args.lectures,
        "stage_order": list(STAGES),
        "plan": plan,
        "results": results,
        "passed": True,
        "started_at": started_at,
        "finished_at": utc_now(),
    }
    write_summary(summary_path, summary)
    print(json.dumps({"completed": True, "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
