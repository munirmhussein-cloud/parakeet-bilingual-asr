
from __future__ import annotations

import argparse
import os
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


VIEWS = (
    "whole",
    "canonical_20s",
    "context_10s_stride_5s",
    "local_2p5s_contiguous",
)


def run(
    command: list[str],
    cwd: Path,
) -> None:

    print("$", " ".join(command), flush=True)

    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def run_reconciliation(
    command: list[str],
    cwd: Path,
) -> int:

    print("$", " ".join(command), flush=True)

    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
    )

    return result.returncode


def count_jsonl_rows(path: Path) -> int:
    count = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                json.loads(line)
                count += 1

    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Silver v3 production construction while "
            "recording, rather than enforcing, reconciliation "
            "quality-gate results."
        )
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--lecture-id",
        required=True,
    )

    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--silver-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--view-workers",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    repo = args.repo_root.resolve()
    silver = args.silver_root.resolve()

    output_dir = silver / "reconciled_fixed"

    output_prefix = (
        f"{args.lecture_id}_silver_v3_fixed"
    )

    silver.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Prepare views
    # --------------------------------------------------------

    run(
        [
            sys.executable,
            "-m",
            "pipeline.silver_v3.prepare_views",
            "--lecture-id",
            args.lecture_id,
            "--audio",
            str(args.audio),
            "--output-root",
            str(silver),
        ],
        repo,
    )

    # --------------------------------------------------------
    # Hosted inference
    #
    # Existing valid segment outputs are skipped by the hosted
    # runner, so this safely resumes partially completed views.
    # --------------------------------------------------------

    # Run independent hosted views concurrently.
    # The worker allocation is weighted toward the larger local
    # and context manifests while keeping the total request
    # budget close to args.view_workers.
    view_names = [
        "whole",
        "canonical_20s",
        "context_10s_stride_5s",
        "local_2p5s_contiguous",
    ]
    
    total_budget = max(4, int(args.view_workers))
    
    # Fixed allocation selected from production timing:
    # context was the critical path at 4 workers, while local
    # completed substantially earlier at 9 workers.
    if total_budget == 16:
        view_worker_counts = {
            "whole": 1,
            "canonical_20s": 2,
            "context_10s_stride_5s": 6,
            "local_2p5s_contiguous": 7,
        }
    else:
        weights = {
            "whole": 0,
            "canonical_20s": 1,
            "context_10s_stride_5s": 3,
            "local_2p5s_contiguous": 4,
        }

        remaining_budget = max(0, total_budget - 4)
        weight_total = sum(weights.values())

        view_worker_counts = {
            view: 1 + (
                round(
                    remaining_budget
                    * weights[view]
                    / weight_total
                )
                if weight_total
                else 0
            )
            for view in view_names
        }
    # Correct rounding so the total never exceeds the budget.
    while sum(view_worker_counts.values()) > total_budget:
        for view in (
            "canonical_20s",
            "context_10s_stride_5s",
            "local_2p5s_contiguous",
        ):
            if (
                sum(view_worker_counts.values())
                <= total_budget
            ):
                break
    
            if view_worker_counts[view] > 1:
                view_worker_counts[view] -= 1
    
    while sum(view_worker_counts.values()) < total_budget:
        view_worker_counts[
            "local_2p5s_contiguous"
        ] += 1
    
    print(
        "Concurrent hosted-view worker allocation:",
        view_worker_counts,
        flush=True,
    )
    
    def run_one_hosted_view(view: str) -> tuple[str, int]:
        manifest = (
            args.silver_root
            / "manifests"
            / f"{args.lecture_id}_{view}.jsonl"
        )
    
        output_dir = (
            args.silver_root
            / "raw_parakeet"
            / view
        )
    
        report = (
            args.silver_root
            / f"silver_v3_{view}_hosted_report.json"
        )
    
        command = [
            sys.executable,
            "-u",
            "-m",
            "pipeline.silver_v3.run_hosted_view",
            "--lecture-id",
            args.lecture_id,
            "--view",
            view,
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report),
            "--workers",
            str(view_worker_counts[view]),
        ]
    
        print(
            "$ " + " ".join(command),
            flush=True,
        )
    
        result = subprocess.run(
            command,
            cwd=str(args.repo_root),
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
            },
            check=False,
        )
    
        return view, result.returncode
    
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=4,
    ) as executor:
        futures = {
            executor.submit(
                run_one_hosted_view,
                view,
            ): view
            for view in view_names
        }
    
        hosted_failures = []
    
        for future in concurrent.futures.as_completed(
            futures
        ):
            view = futures[future]
    
            try:
                completed_view, return_code = (
                    future.result()
                )
            except Exception as error:
                hosted_failures.append(
                    (view, repr(error))
                )
                print(
                    f"Hosted view {view} raised: {error}",
                    flush=True,
                )
                continue
    
            print(
                f"Hosted view {completed_view} "
                f"finished with code {return_code}",
                flush=True,
            )
    
            if return_code != 0:
                hosted_failures.append(
                    (
                        completed_view,
                        f"return_code={return_code}",
                    )
                )
    
    if hosted_failures:
        raise RuntimeError(
            "Hosted view failures: "
            + repr(hosted_failures)
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalization_report = (
        silver
        / "silver_v3_normalization_report.json"
    )

    run(
        [
            sys.executable,
            "-m",
            "pipeline.silver_v3.normalize_views",
            "--lecture-id",
            args.lecture_id,
            "--silver-root",
            str(silver),
            "--report",
            str(normalization_report),
        ],
        repo,
    )

    # --------------------------------------------------------
    # Reconcile
    #
    # The reconciler writes all artifacts before returning its
    # quality-gate status. A nonzero code is therefore recorded
    # rather than automatically treated as construction failure.
    # --------------------------------------------------------

    normalized = silver / "normalized"

    canonical_manifest = (
        silver
        / "manifests"
        / f"{args.lecture_id}_canonical_20s.jsonl"
    )

    reconciliation_command = [
        sys.executable,
        str(
            repo
            / "scripts"
            / "reconcile_silver_v3_lattice.py"
        ),
        "--lecture-id",
        args.lecture_id,
        "--whole",
        str(
            normalized
            / f"{args.lecture_id}_whole_normalized.jsonl"
        ),
        "--canonical",
        str(
            normalized
            / f"{args.lecture_id}_canonical_20s_normalized.jsonl"
        ),
        "--context",
        str(
            normalized
            / f"{args.lecture_id}_context_10s_stride_5s_normalized.jsonl"
        ),
        "--local",
        str(
            normalized
            / f"{args.lecture_id}_local_2p5s_contiguous_normalized.jsonl"
        ),
        "--canonical-manifest",
        str(canonical_manifest),
        "--output-dir",
        str(output_dir),
        "--output-prefix",
        output_prefix,
        "--schema-version",
        "silver_v3_segment_level_v2",
        "--title",
        f"{args.lecture_id} Silver v3",
    ]

    reconciliation_return_code = run_reconciliation(
        reconciliation_command,
        repo,
    )

    # --------------------------------------------------------
    # Required construction artifacts
    # --------------------------------------------------------

    segment_path = (
        output_dir
        / f"{output_prefix}_segment_level.jsonl"
    )

    provenance_path = (
        output_dir
        / f"{output_prefix}_token_provenance.jsonl"
    )

    json_path = (
        output_dir
        / f"{output_prefix}.json"
    )

    docx_path = (
        output_dir
        / f"{output_prefix}.docx"
    )

    required_outputs = [
        segment_path,
        provenance_path,
        json_path,
        docx_path,
        normalization_report,
    ]

    missing = [
        path
        for path in required_outputs
        if (
            not path.is_file()
            or path.stat().st_size == 0
        )
    ]

    if missing:
        raise FileNotFoundError(
            "Silver v3 construction outputs are missing:\n"
            + "\n".join(str(path) for path in missing)
        )

    segment_count = count_jsonl_rows(
        segment_path
    )

    if segment_count == 0:
        raise RuntimeError(
            "Silver v3 segment JSONL contains no rows."
        )

    # --------------------------------------------------------
    # Capture reconciliation validation as metadata
    # --------------------------------------------------------

    reconciliation_payload = json.loads(
        json_path.read_text(encoding="utf-8")
    )

    reconciliation_validation = (
        reconciliation_payload.get(
            "validation",
            {}
        )
        if isinstance(
            reconciliation_payload,
            dict,
        )
        else {}
    )

    quality_gate_passed = bool(
        reconciliation_validation.get(
            "passed",
            reconciliation_return_code == 0,
        )
    )

    report_path = (
        output_dir
        / f"{output_prefix}_production_run_report.json"
    )

    report = {
        "schema_version":
        "silver_v3_production_run_report_v2",
        "lecture_id": args.lecture_id,
        "segment_jsonl": str(segment_path),
        "token_provenance_jsonl":
        str(provenance_path),
        "json": str(json_path),
        "docx": str(docx_path),
        "segment_count": segment_count,
        "construction_completed": True,
        "reconciliation_return_code":
        reconciliation_return_code,
        "quality_gate_passed":
        quality_gate_passed,
        "quality_gate_recorded_not_enforced": True,
        "reconciliation_validation":
        reconciliation_validation,
        "evaluation_performed": False,
        "validation_performed": True,
    }

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "completed": True,
                "lecture_id":
                args.lecture_id,
                "segment_count":
                segment_count,
                "construction_completed":
                True,
                "quality_gate_passed":
                quality_gate_passed,
                "reconciliation_return_code":
                reconciliation_return_code,
                "segment_jsonl":
                str(segment_path),
                "production_report":
                str(report_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
