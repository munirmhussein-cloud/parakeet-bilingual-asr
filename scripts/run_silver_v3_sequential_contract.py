from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from finalize_silver_v3_contract import (
    EXPECTED_QUALITY_SCHEMA,
    EXPECTED_SCHEMA,
    validate_archive,
)


VIEWS = (
    "whole",
    "canonical_20s",
    "context_10s_stride_5s",
    "local_2p5s_contiguous",
)
SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".opus",
}
SEERAH_RE = re.compile(r"^seerah\D*0*(\d{1,3})(?!\d)", re.IGNORECASE)
LECTURE_RE = re.compile(r"^lecture[_\-\s]*0*(\d{1,3})(?!\d)", re.IGNORECASE)
SAMPLE_RATE = 16_000
PRINT_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read JSONL: {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise RuntimeError(f"JSONL contains no rows: {path}")
    return rows


def git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def stream(command: list[str], cwd: Path, prefix: str) -> int:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with PRINT_LOCK:
        print(f"[{prefix}] $ {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        if text:
            with PRINT_LOCK:
                print(f"[{prefix}] {text}", flush=True)
    return process.wait()


def parse_lecture_number(path: Path) -> int | None:
    for pattern in (SEERAH_RE, LECTURE_RE):
        match = pattern.match(path.stem)
        if match:
            return int(match.group(1))
    return None


def discover_audio(audio_root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(audio_root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        number = parse_lecture_number(path)
        if number is None:
            continue
        grouped.setdefault(f"lecture_{number:03d}", []).append(path.resolve())
    duplicates = {lecture: paths for lecture, paths in grouped.items() if len(paths) != 1}
    unique = {lecture: paths[0] for lecture, paths in grouped.items() if len(paths) == 1}
    return unique, duplicates


def require_runtime(repo_root: Path, drive_root: Path) -> None:
    if not drive_root.is_dir():
        raise FileNotFoundError(
            f"Google Drive production root is not mounted or missing: {drive_root}"
        )
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is not configured.")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    gpu_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not gpu_names or not any("A100" in name for name in gpu_names):
        raise RuntimeError(f"NVIDIA A100 is required; detected: {gpu_names}")
    for required in (
        repo_root / "pipeline/silver_v3/prepare_views.py",
        repo_root / "pipeline/silver_v3/run_hosted_view.py",
        repo_root / "pipeline/silver_v3/finalize_fixed.py",
        repo_root / "scripts/finalize_silver_v3_contract.py",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)


def window_spans(
    duration_seconds: float,
    *,
    window_seconds: float,
    stride_seconds: float,
) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_seconds - 1e-9:
        end = min(start + window_seconds, duration_seconds)
        if end <= start:
            break
        spans.append((round(start, 6), round(end, 6)))
        start += stride_seconds
    return spans


def expected_spans(duration: float) -> dict[str, list[tuple[float, float]]]:
    return {
        "whole": [(0.0, round(duration, 6))],
        "canonical_20s": window_spans(
            duration, window_seconds=20.0, stride_seconds=20.0
        ),
        "context_10s_stride_5s": window_spans(
            duration, window_seconds=10.0, stride_seconds=5.0
        ),
        "local_2p5s_contiguous": window_spans(
            duration, window_seconds=2.5, stride_seconds=2.5
        ),
    }


def validate_manifest_identity(
    manifest_path: Path,
    *,
    lecture_id: str,
    view: str,
    spans: list[tuple[float, float]],
    silver_root: Path,
) -> list[dict[str, Any]]:
    rows = read_jsonl(manifest_path)
    if len(rows) != len(spans):
        raise RuntimeError(
            f"Manifest row-count mismatch for {view}: {len(rows)} != {len(spans)}"
        )
    for position, (row, (start, end)) in enumerate(zip(rows, spans)):
        expected_id = f"{lecture_id}__{view}__{position:05d}"
        expected_audio = silver_root / "audio_views" / view / f"{expected_id}.wav"
        checks = {
            "lecture_id": (row.get("lecture_id"), lecture_id),
            "view": (row.get("view"), view),
            "segment_id": (row.get("segment_id"), expected_id),
            "segment_position": (row.get("segment_position"), position),
            "global_start": (float(row.get("global_start", -1)), start),
            "global_end": (float(row.get("global_end", -1)), end),
            "audio_filepath": (
                str(Path(str(row.get("audio_filepath", ""))).resolve()),
                str(expected_audio.resolve()),
            ),
        }
        mismatches = {
            key: {"observed": observed, "expected": expected}
            for key, (observed, expected) in checks.items()
            if observed != expected
        }
        if mismatches:
            raise RuntimeError(
                f"Manifest identity mismatch at {manifest_path}:{position + 1}: "
                + json.dumps(mismatches, sort_keys=True)
            )
    return rows


def validate_partial_preparation(
    *,
    audio_path: Path,
    source_sha256: str,
    silver_root: Path,
    lecture_id: str,
) -> dict[str, Any]:
    preparation_report = silver_root / "silver_v3_multiview_preparation_report.json"
    if preparation_report.is_file():
        report = read_json(preparation_report)
        if report.get("lecture_id") != lecture_id:
            raise RuntimeError("Preparation report lecture identity mismatch")
        if report.get("source_sha256") != source_sha256:
            raise RuntimeError(
                "Existing Silver v3 preparation belongs to different source audio"
            )
        if report.get("passed") is not True:
            raise RuntimeError("Existing Silver v3 preparation report has passed=false")
        return {
            "mode": "validated_preparation_report",
            "report": str(preparation_report),
            "source_sha256": source_sha256,
        }

    manifests_root = silver_root / "manifests"
    audio_views_root = silver_root / "audio_views"
    has_partial = (
        manifests_root.exists()
        or audio_views_root.exists()
        or (silver_root / "raw_parakeet").exists()
        or (silver_root / "normalized").exists()
        or (silver_root / "reconciled_fixed").exists()
    )
    if not has_partial:
        return {"mode": "no_existing_preparation", "source_sha256": source_sha256}

    # A preparation report is written only after all views finish. For an
    # interrupted preparation, validate every existing WAV sample against the
    # selected source before allowing prepare_views.py to fill missing files.
    try:
        import librosa
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "librosa, numpy, and soundfile are required to validate partial Silver v3 audio views"
        ) from exc

    waveform, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
        raise RuntimeError("Selected source audio decoded to an invalid waveform")
    duration = len(waveform) / sample_rate
    spans_by_view = expected_spans(duration)
    validated_wavs = 0

    expected_manifest_names = {
        f"{lecture_id}_{view}.jsonl" for view in VIEWS
    } | {f"{lecture_id}_canonical_manifest.jsonl"}
    if manifests_root.is_dir():
        foreign_manifests = sorted(
            path.name
            for path in manifests_root.glob("*.jsonl")
            if path.name not in expected_manifest_names
        )
        if foreign_manifests:
            raise RuntimeError(
                f"Manifest directory contains foreign lecture files: {foreign_manifests[:20]}"
            )

    for view, spans in spans_by_view.items():
        expected_ids = {
            f"{lecture_id}__{view}__{position:05d}": (position, start, end)
            for position, (start, end) in enumerate(spans)
        }
        manifest_path = manifests_root / f"{lecture_id}_{view}.jsonl"
        if manifest_path.exists():
            validate_manifest_identity(
                manifest_path,
                lecture_id=lecture_id,
                view=view,
                spans=spans,
                silver_root=silver_root,
            )

        view_dir = audio_views_root / view
        existing_wavs = sorted(view_dir.glob("*.wav")) if view_dir.is_dir() else []
        observed_ids = {path.stem for path in existing_wavs}
        foreign_ids = sorted(observed_ids - set(expected_ids))
        if foreign_ids:
            raise RuntimeError(
                f"Existing {view} audio contains foreign lecture/view IDs: {foreign_ids[:20]}"
            )
        if manifest_path.exists() and observed_ids != set(expected_ids):
            missing = sorted(set(expected_ids) - observed_ids)
            raise RuntimeError(
                f"Manifest exists but {view} audio files are incomplete: {missing[:20]}"
            )

        for wav_path in existing_wavs:
            _, start, end = expected_ids[wav_path.stem]
            start_sample = max(0, min(int(round(start * sample_rate)), len(waveform)))
            end_sample = max(
                start_sample,
                min(int(round(end * sample_rate)), len(waveform)),
            )
            expected = waveform[start_sample:end_sample]
            info = sf.info(wav_path)
            if info.samplerate != sample_rate or info.channels != 1 or info.frames != len(expected):
                raise RuntimeError(f"Existing audio-view shape mismatch: {wav_path}")
            observed, observed_rate = sf.read(
                wav_path, dtype="float32", always_2d=False
            )
            observed = np.asarray(observed, dtype=np.float32)
            if observed_rate != sample_rate or observed.shape != expected.shape:
                raise RuntimeError(f"Existing audio-view decode mismatch: {wav_path}")
            maximum_error = (
                float(np.max(np.abs(observed - expected))) if len(expected) else 0.0
            )
            if maximum_error > (2.0 / 32768.0):
                raise RuntimeError(
                    f"Existing audio view does not match selected source: {wav_path}; "
                    f"maximum_sample_error={maximum_error}"
                )
            validated_wavs += 1

    canonical_alias = manifests_root / f"{lecture_id}_canonical_manifest.jsonl"
    canonical_manifest = manifests_root / f"{lecture_id}_canonical_20s.jsonl"
    if canonical_alias.exists():
        alias_rows = read_jsonl(canonical_alias)
        canonical_rows = read_jsonl(canonical_manifest)
        if alias_rows != canonical_rows:
            raise RuntimeError("Canonical manifest alias differs from canonical_20s manifest")

    return {
        "mode": "validated_interrupted_preparation",
        "source_sha256": source_sha256,
        "validated_existing_wav_count": validated_wavs,
        "duration_seconds": round(duration, 6),
    }


def allocate_view_workers(total_budget: int) -> dict[str, int]:
    total_budget = max(4, int(total_budget))
    if total_budget == 16:
        return {
            "whole": 1,
            "canonical_20s": 2,
            "context_10s_stride_5s": 6,
            "local_2p5s_contiguous": 7,
        }
    weights = {
        "whole": 0,
        "canonical_20s": 1,
        "context_10s_stride_5s": 3,
        "local_2p5s_contiguous": 4,
    }
    remaining = max(0, total_budget - 4)
    weight_total = sum(weights.values())
    allocation = {
        view: 1 + round(remaining * weights[view] / weight_total)
        for view in VIEWS
    }
    while sum(allocation.values()) > total_budget:
        for view in ("canonical_20s", "context_10s_stride_5s", "local_2p5s_contiguous"):
            if sum(allocation.values()) <= total_budget:
                break
            if allocation[view] > 1:
                allocation[view] -= 1
    while sum(allocation.values()) < total_budget:
        allocation["local_2p5s_contiguous"] += 1
    return allocation


def run_hosted_views(
    *,
    repo_root: Path,
    silver_root: Path,
    lecture_id: str,
    view_workers: int,
    max_attempts: int,
) -> dict[str, Any]:
    allocation = allocate_view_workers(view_workers)
    print(f"[{lecture_id}] Hosted worker allocation: {allocation}", flush=True)

    def run_view(view: str) -> tuple[str, int, int]:
        command = [
            sys.executable,
            "-u",
            "-m",
            "pipeline.silver_v3.run_hosted_view",
            "--lecture-id",
            lecture_id,
            "--view",
            view,
            "--manifest",
            str(silver_root / "manifests" / f"{lecture_id}_{view}.jsonl"),
            "--output-dir",
            str(silver_root / "raw_parakeet" / view),
            "--report",
            str(silver_root / f"silver_v3_{view}_hosted_report.json"),
            "--workers",
            str(allocation[view]),
        ]
        last_code = 1
        for attempt in range(1, max_attempts + 1):
            last_code = stream(command, repo_root, f"{lecture_id}:{view}:attempt{attempt}")
            if last_code == 0:
                return view, last_code, attempt
            if attempt < max_attempts:
                with PRINT_LOCK:
                    print(
                        f"[{lecture_id}:{view}] retrying; valid existing outputs will be reused",
                        flush=True,
                    )
        return view, last_code, max_attempts

    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_view, view): view for view in VIEWS}
        for future in concurrent.futures.as_completed(futures):
            view = futures[future]
            try:
                completed_view, code, attempts = future.result()
            except Exception as exc:
                raise RuntimeError(f"Hosted view {view} raised: {exc}") from exc
            results[completed_view] = {
                "return_code": code,
                "attempts": attempts,
                "workers": allocation[completed_view],
            }
            if code != 0:
                raise RuntimeError(
                    f"Hosted view {completed_view} failed after {attempts} attempts"
                )
    return {"allocation": allocation, "views": results}


def package_matches_current_sources(
    validation: dict[str, Any], silver_root: Path
) -> None:
    manifest = validation["manifest"]
    reconciled = silver_root / "reconciled_fixed"
    for item in manifest["files"]:
        filename = item["filename"]
        source = (
            silver_root / "silver_v3_normalization_report.json"
            if filename == "silver_v3_normalization_report.json"
            else reconciled / filename
        )
        if not source.is_file() or source.stat().st_size != item["size_bytes"]:
            raise RuntimeError(f"Package source file is missing or changed: {source}")
        if sha256_file(source) != item["sha256"]:
            raise RuntimeError(f"Package source file hash changed: {source}")


def find_valid_package(
    *,
    silver_root: Path,
    lecture_id: str,
) -> tuple[Path | None, dict[str, Any] | None]:
    final_package = silver_root / "final_package"
    candidates = (
        sorted(
            final_package.glob(f"{lecture_id}_silver_v3_repaired_*.zip"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if final_package.is_dir()
        else []
    )
    failures: list[str] = []
    for candidate in candidates:
        try:
            validation = validate_archive(candidate, lecture_id)
            package_matches_current_sources(validation, silver_root)
            return candidate, validation
        except Exception as exc:
            failures.append(f"{candidate}: {exc}")
    if failures:
        raise RuntimeError(
            "Existing repaired package candidate failed strict validation:\n"
            + "\n".join(failures)
        )
    return None, None


def backup_existing_marker(marker_path: Path) -> Path | None:
    if not marker_path.exists():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = marker_path.with_name(f".silver_v3_complete.stale_{timestamp}.json")
    counter = 1
    while backup.exists():
        backup = marker_path.with_name(
            f".silver_v3_complete.stale_{timestamp}_{counter}.json"
        )
        counter += 1
    os.replace(marker_path, backup)
    return backup


def marker_matches(
    marker_path: Path,
    *,
    lecture_id: str,
    source_audio: Path,
    source_sha256: str,
    validation: dict[str, Any],
) -> bool:
    if not marker_path.is_file():
        return False
    try:
        marker = read_json(marker_path)
    except Exception:
        return False
    return (
        marker.get("schema_version") == "silver_v3_completion_marker_v1"
        and marker.get("status") == "completed"
        and marker.get("lecture_id") == lecture_id
        and marker.get("contract_schema") == EXPECTED_SCHEMA
        and marker.get("quality_validation_schema") == EXPECTED_QUALITY_SCHEMA
        and marker.get("source_audio") == str(source_audio)
        and marker.get("source_audio_sha256") == source_sha256
        and marker.get("package_sha256") == validation["archive_sha256"]
        and marker.get("repository_commit") == validation["repository_commit"]
    )


def write_completion_marker(
    marker_path: Path,
    *,
    lecture_id: str,
    silver_root: Path,
    source_audio: Path,
    source_sha256: str,
    validation: dict[str, Any],
) -> dict[str, Any]:
    marker = {
        "schema_version": "silver_v3_completion_marker_v1",
        "status": "completed",
        "lecture_id": lecture_id,
        "contract_schema": EXPECTED_SCHEMA,
        "quality_validation_schema": EXPECTED_QUALITY_SCHEMA,
        "repository_commit": validation["repository_commit"],
        "source_audio": str(source_audio),
        "source_audio_size_bytes": source_audio.stat().st_size,
        "source_audio_sha256": source_sha256,
        "silver_root": str(silver_root),
        "package_archive": validation["archive"],
        "package_checksum_file": validation["archive_checksum_file"],
        "package_sha256": validation["archive_sha256"],
        "package_root": validation["package_root"],
        "zip_member_count": validation["member_count"],
        "segment_count": validation["segment_count"],
        "token_provenance_row_count": validation["token_provenance_row_count"],
        "total_tokens": validation["total_tokens"],
        "quality_gates": validation["quality_gates"],
        "quality_thresholds": validation["quality_thresholds"],
        "finished_at": utc_now(),
    }
    atomic_write_json(marker_path, marker)
    if not marker_matches(
        marker_path,
        lecture_id=lecture_id,
        source_audio=source_audio,
        source_sha256=source_sha256,
        validation=validation,
    ):
        raise RuntimeError("Completion marker did not validate after atomic write")
    return marker


def immutable_title(lecture_id: str) -> str:
    return f"Lecture {lecture_id.rsplit('_', 1)[-1]} — Silver v3 Repaired Multiview Transcript"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Silver v3 strictly one lecture at a time. Completion is accepted "
            "only after the immutable Lecture 001 repair, unchanged quality gates, "
            "repaired package, internal hashes, and external ZIP checksum validate."
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Rebuild repaired outputs/package while preserving valid hosted inference. "
            "This never deletes the Silver v3 tree."
        ),
    )
    args = parser.parse_args()

    if not 1 <= args.start <= args.end <= 104:
        raise ValueError("Lecture range must satisfy 1 <= start <= end <= 104")
    if args.view_workers < 4:
        raise ValueError("--view-workers must be at least 4")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")

    repo_root = args.repo_root.resolve()
    drive_root = args.drive_root.resolve()
    audio_root = drive_root / "audio"
    lectures_root = drive_root / "production_a100" / "lectures"
    state_path = (
        drive_root
        / "production_a100"
        / "pipeline_state"
        / "silver_v3_sequential_contract_summary.json"
    )
    require_runtime(repo_root, drive_root)
    if not audio_root.is_dir():
        raise FileNotFoundError(audio_root)

    audio_map, duplicates = discover_audio(audio_root)
    requested_ids = [f"lecture_{number:03d}" for number in range(args.start, args.end + 1)]
    ambiguous = {lecture: duplicates[lecture] for lecture in requested_ids if lecture in duplicates}
    if ambiguous:
        details = "\n\n".join(
            lecture + ":\n" + "\n".join(f"  - {path}" for path in paths)
            for lecture, paths in sorted(ambiguous.items())
        )
        raise RuntimeError("Ambiguous source audio; refusing to continue:\n\n" + details)
    missing_audio = [lecture for lecture in requested_ids if lecture not in audio_map]
    if missing_audio:
        raise FileNotFoundError(
            "Missing source audio for requested lectures:\n" + "\n".join(missing_audio)
        )

    commit = git_head(repo_root)
    state: dict[str, Any] = {
        "schema_version": "silver_v3_sequential_contract_summary_v2",
        "contract_reference": "lecture_001_silver_v3_repaired_export_package_v1",
        "repository_commit": commit,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "finished_at": None,
        "completed": False,
        "results": {},
    }
    atomic_write_json(state_path, state)

    print("=" * 100)
    print("SILVER v3 STRICT SEQUENTIAL PRODUCTION")
    print("=" * 100)
    print("Repository commit:", commit)
    print("Lectures:", f"{requested_ids[0]}–{requested_ids[-1]}")
    print("View workers:", args.view_workers)
    print("NVIDIA_API_KEY configured: yes")
    print("State:", state_path)

    if args.plan_only:
        for index, lecture_id in enumerate(requested_ids, start=1):
            print(f"{index:>3}/{len(requested_ids)} {lecture_id} | {audio_map[lecture_id].name}")
        state["completed"] = True
        state["finished_at"] = utc_now()
        state["updated_at"] = utc_now()
        atomic_write_json(state_path, state)
        return 0

    current_lecture = "unknown"
    try:
        for index, lecture_id in enumerate(requested_ids, start=1):
            current_lecture = lecture_id
            started = time.monotonic()
            audio_path = audio_map[lecture_id]
            source_sha = sha256_file(audio_path)
            silver_root = lectures_root / lecture_id / "silver_v3"
            silver_root.mkdir(parents=True, exist_ok=True)
            marker_path = silver_root / ".silver_v3_complete.json"
            result: dict[str, Any] = {
                "status": "running",
                "lecture_id": lecture_id,
                "source_audio": str(audio_path),
                "source_audio_sha256": source_sha,
                "started_at": utc_now(),
                "stages": {},
            }
            state["results"][lecture_id] = result
            state["updated_at"] = utc_now()
            atomic_write_json(state_path, state)

            print("\n" + "=" * 100)
            print(f"[{index}/{len(requested_ids)}] {lecture_id} | {audio_path.name}")
            print("=" * 100)

            preparation_audit = validate_partial_preparation(
                audio_path=audio_path,
                source_sha256=source_sha,
                silver_root=silver_root,
                lecture_id=lecture_id,
            )
            result["stages"]["source_and_partial_artifact_audit"] = preparation_audit

            if not args.overwrite:
                package_path, package_validation = find_valid_package(
                    silver_root=silver_root,
                    lecture_id=lecture_id,
                )
                if package_path is not None and package_validation is not None:
                    if not marker_matches(
                        marker_path,
                        lecture_id=lecture_id,
                        source_audio=audio_path,
                        source_sha256=source_sha,
                        validation=package_validation,
                    ):
                        backup = backup_existing_marker(marker_path)
                        marker = write_completion_marker(
                            marker_path,
                            lecture_id=lecture_id,
                            silver_root=silver_root,
                            source_audio=audio_path,
                            source_sha256=source_sha,
                            validation=package_validation,
                        )
                    else:
                        backup = None
                        marker = read_json(marker_path)
                    result.update(
                        {
                            "status": "skipped_contract_valid",
                            "finished_at": utc_now(),
                            "wall_seconds": round(time.monotonic() - started, 3),
                            "package": package_validation,
                            "completion_marker": marker,
                            "stale_marker_backup": str(backup) if backup else None,
                        }
                    )
                    state["results"][lecture_id] = result
                    state["updated_at"] = utc_now()
                    atomic_write_json(state_path, state)
                    print(f"SKIPPED {lecture_id}: repaired package and marker are contract-valid")
                    continue

            stale_marker = backup_existing_marker(marker_path)
            result["stale_marker_backup"] = str(stale_marker) if stale_marker else None

            prepare_command = [
                sys.executable,
                "-u",
                "-m",
                "pipeline.silver_v3.prepare_views",
                "--lecture-id",
                lecture_id,
                "--audio",
                str(audio_path),
                "--output-root",
                str(silver_root),
            ]
            prepare_code = stream(prepare_command, repo_root, f"{lecture_id}:prepare")
            result["stages"]["prepare_views"] = {"return_code": prepare_code}
            state["updated_at"] = utc_now()
            atomic_write_json(state_path, state)
            if prepare_code != 0:
                raise RuntimeError("View preparation failed")
            prep_report = read_json(silver_root / "silver_v3_multiview_preparation_report.json")
            if (
                prep_report.get("passed") is not True
                or prep_report.get("lecture_id") != lecture_id
                or prep_report.get("source_sha256") != source_sha
            ):
                raise RuntimeError("View preparation report failed identity/quality validation")

            hosted = run_hosted_views(
                repo_root=repo_root,
                silver_root=silver_root,
                lecture_id=lecture_id,
                view_workers=args.view_workers,
                max_attempts=args.max_attempts,
            )
            result["stages"]["hosted_views"] = hosted
            state["updated_at"] = utc_now()
            atomic_write_json(state_path, state)

            repair_command = [
                sys.executable,
                "-u",
                str(repo_root / "pipeline/silver_v3/finalize_fixed.py"),
                "--repo-root",
                str(repo_root),
                "--silver-root",
                str(silver_root),
                "--lecture-id",
                lecture_id,
                "--schema-version",
                "silver_v3_segment_level_v4",
                "--title",
                immutable_title(lecture_id),
                "--min-a-tier-ratio",
                "0.60",
                "--max-repeated-6grams",
                "1",
                "--pilot-segments",
                "30",
                "--pilot-min-a-tier-ratio",
                "0.75",
                "--pilot-max-repeated-6grams",
                "1",
            ]
            repair_code = stream(repair_command, repo_root, f"{lecture_id}:repair")
            result["stages"]["repair_finalization"] = {"return_code": repair_code}
            state["updated_at"] = utc_now()
            atomic_write_json(state_path, state)
            if repair_code != 0:
                raise RuntimeError("Repair/finalization failed unchanged quality gates")

            finalization_report = read_json(
                silver_root
                / "reconciled_fixed"
                / f"{lecture_id}_silver_v3_fixed_finalization_report.json"
            )
            if finalization_report.get("passed") is not True:
                raise RuntimeError("Repair finalization report has passed=false")

            package_command = [
                sys.executable,
                "-u",
                str(repo_root / "scripts/finalize_silver_v3_contract.py"),
                "--repo-root",
                str(repo_root),
                "--silver-root",
                str(silver_root),
                "--lecture-id",
                lecture_id,
                "--copy-to",
                str(silver_root / "final_package"),
            ]
            package_code = stream(package_command, repo_root, f"{lecture_id}:package")
            result["stages"]["quality_package_and_strict_validation"] = {
                "return_code": package_code
            }
            state["updated_at"] = utc_now()
            atomic_write_json(state_path, state)
            if package_code != 0:
                raise RuntimeError("Repaired package creation or strict validation failed")

            package_path = (
                silver_root
                / "final_package"
                / f"{lecture_id}_silver_v3_repaired_{commit[:8]}.zip"
            )
            validation = validate_archive(
                package_path,
                lecture_id,
                expected_repository_commit=commit,
            )
            package_matches_current_sources(validation, silver_root)
            marker = write_completion_marker(
                marker_path,
                lecture_id=lecture_id,
                silver_root=silver_root,
                source_audio=audio_path,
                source_sha256=source_sha,
                validation=validation,
            )
            result.update(
                {
                    "status": "completed_contract_valid",
                    "finished_at": utc_now(),
                    "wall_seconds": round(time.monotonic() - started, 3),
                    "package": validation,
                    "completion_marker": marker,
                }
            )
            state["results"][lecture_id] = result
            state["updated_at"] = utc_now()
            atomic_write_json(state_path, state)
            print(
                f"COMPLETED {lecture_id} | segments={validation['segment_count']} | "
                f"tokens={validation['total_tokens']} | sha256={validation['archive_sha256']}",
                flush=True,
            )

    except KeyboardInterrupt:
        state["interrupted"] = True
        state["updated_at"] = utc_now()
        state["finished_at"] = utc_now()
        atomic_write_json(state_path, state)
        print(f"INTERRUPTED at {current_lecture}; no unvalidated completion marker was written")
        return 130
    except Exception as exc:
        silver_root = lectures_root / current_lecture / "silver_v3"
        (silver_root / ".silver_v3_complete.json").unlink(missing_ok=True)
        existing = state["results"].get(current_lecture, {})
        existing.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": str(exc),
            }
        )
        state["results"][current_lecture] = existing
        state["updated_at"] = utc_now()
        state["finished_at"] = utc_now()
        atomic_write_json(state_path, state)
        print(f"FAILED {current_lecture}: {exc}", flush=True)
        return 1

    state["finished_at"] = utc_now()
    state["updated_at"] = utc_now()
    state["completed"] = True
    atomic_write_json(state_path, state)
    print("\nAll requested lectures satisfy the immutable Lecture 001 Silver v3 contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
