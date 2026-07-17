from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

STAGES = ("bronze", "silver", "silver_plus")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render(value: str, context: dict[str, str]) -> str:
    return value.format(**context)


def run_one(
    stage: str,
    lecture_id: str,
    stage_config: dict[str, Any],
    common: dict[str, str],
    log_root: Path,
    resume: bool,
) -> dict[str, Any]:
    context = {**common, "lecture_id": lecture_id, "stage": stage}
    expected = [Path(render(path, context)) for path in stage_config.get("expected_outputs", [])]
    if resume and expected and all(path.exists() for path in expected):
        return {"lecture_id": lecture_id, "stage": stage, "status": "skipped", "reason": "outputs_exist"}

    command = [render(str(part), context) for part in stage_config["command"]]
    env = os.environ.copy()
    env.update({key: render(str(value), context) for key, value in stage_config.get("env", {}).items()})
    cwd = Path(render(stage_config.get("cwd", common["repo_root"]), context))
    log_dir = log_root / stage
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{lecture_id}.log"

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
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
            f"{stage} failed for {lecture_id} with return code {result.returncode}; log={log_path}"
        )

    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(
            f"{stage} completed for {lecture_id} but expected outputs are missing: {missing}"
        )

    return {"lecture_id": lecture_id, "stage": stage, "status": "completed", "log": str(log_path)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Bronze -> Silver -> Silver+ with a barrier between stages and parallel lectures within each stage."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lectures", nargs="+", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    config = load_json(args.config)
    common = {key: str(value) for key, value in config["common"].items()}
    log_root = Path(common.get("log_root", Path(common["drive_root"]) / "pipeline_logs"))
    results: list[dict[str, Any]] = []

    for stage in STAGES:
        stage_config = config["stages"][stage]
        max_workers = max(1, int(stage_config.get("max_workers", 1)))
        print(f"\n=== {stage.upper()} | lectures={len(args.lectures)} | workers={max_workers} ===")
        stage_results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
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
            for future in as_completed(futures):
                result = future.result()
                stage_results.append(result)
                print(json.dumps(result, sort_keys=True))
        results.extend(sorted(stage_results, key=lambda item: item["lecture_id"]))
        print(f"=== {stage.upper()} COMPLETE ===")

    summary = {
        "schema_version": "staged_bronze_silver_silver_plus_batch_v1",
        "lectures": args.lectures,
        "stage_order": list(STAGES),
        "results": results,
        "passed": True,
    }
    summary_path = args.summary or (log_root / "latest_batch_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"completed": True, "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
