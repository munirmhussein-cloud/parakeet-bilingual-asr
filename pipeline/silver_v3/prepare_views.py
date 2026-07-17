from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SAMPLE_RATE = 16_000

VIEW_SPECS = {
    "whole": (None, None),
    "canonical_20s": (20.0, 20.0),
    "context_10s_stride_5s": (10.0, 5.0),
    "local_2p5s_contiguous": (2.5, 2.5),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def window_spans(
    duration_seconds: float,
    *,
    window_seconds: float,
    stride_seconds: float,
) -> list[tuple[float, float]]:
    spans = []
    start = 0.0

    while start < duration_seconds - 1e-9:
        end = min(start + window_seconds, duration_seconds)

        if end <= start:
            break

        spans.append(
            (
                round(start, 6),
                round(end, 6),
            )
        )

        start += stride_seconds

    return spans


def write_view(
    *,
    lecture_id: str,
    view_name: str,
    spans: list[tuple[float, float]],
    waveform: np.ndarray,
    sample_rate: int,
    audio_root: Path,
) -> list[dict]:
    output_dir = audio_root / view_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for position, (global_start, global_end) in enumerate(spans):
        segment_id = (
            f"{lecture_id}__{view_name}__{position:05d}"
        )

        output_path = output_dir / f"{segment_id}.wav"

        start_sample = max(
            0,
            min(
                int(round(global_start * sample_rate)),
                len(waveform),
            ),
        )

        end_sample = max(
            start_sample,
            min(
                int(round(global_end * sample_rate)),
                len(waveform),
            ),
        )

        segment = waveform[start_sample:end_sample]

        if not output_path.exists():
            sf.write(
                output_path,
                segment,
                sample_rate,
                subtype="PCM_16",
            )

        rows.append(
            {
                "schema_version":
                "silver_v3_view_manifest_row_v1",
                "lecture_id":
                lecture_id,
                "view":
                view_name,
                "segment_id":
                segment_id,
                "segment_position":
                position,
                "audio_filepath":
                str(output_path),
                "global_start":
                round(global_start, 6),
                "global_end":
                round(global_end, 6),
                "offset":
                round(global_start, 6),
                "duration":
                round(len(segment) / sample_rate, 6),
                "sample_rate":
                sample_rate,
                "num_samples":
                len(segment),
            }
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare deterministic Silver v3 whole, canonical, "
            "context and local audio views."
        )
    )

    parser.add_argument("--lecture-id", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(args.audio)

    started = time.perf_counter()

    audio_root = args.output_root / "audio_views"
    manifest_root = args.output_root / "manifests"

    waveform, sample_rate = librosa.load(
        args.audio,
        sr=args.sample_rate,
        mono=True,
    )

    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.ndim != 1 or not len(waveform):
        raise ValueError("Decoded waveform is empty or non-mono.")

    if not np.isfinite(waveform).all():
        raise ValueError("Decoded waveform contains non-finite values.")

    duration = len(waveform) / sample_rate

    spans = {
        "whole": [(0.0, round(duration, 6))],
        "canonical_20s": window_spans(
            duration,
            window_seconds=20.0,
            stride_seconds=20.0,
        ),
        "context_10s_stride_5s": window_spans(
            duration,
            window_seconds=10.0,
            stride_seconds=5.0,
        ),
        "local_2p5s_contiguous": window_spans(
            duration,
            window_seconds=2.5,
            stride_seconds=2.5,
        ),
    }

    view_rows = {}

    for view_name, view_spans in spans.items():
        rows = write_view(
            lecture_id=args.lecture_id,
            view_name=view_name,
            spans=view_spans,
            waveform=waveform,
            sample_rate=sample_rate,
            audio_root=audio_root,
        )

        view_rows[view_name] = rows

        write_jsonl(
            manifest_root
            / f"{args.lecture_id}_{view_name}.jsonl",
            rows,
        )

    canonical_manifest = (
        manifest_root
        / f"{args.lecture_id}_canonical_manifest.jsonl"
    )

    write_jsonl(
        canonical_manifest,
        view_rows["canonical_20s"],
    )

    validation = {}

    for view_name, rows in view_rows.items():
        positions = [
            row["segment_position"]
            for row in rows
        ]

        ids = [
            row["segment_id"]
            for row in rows
        ]

        missing_audio = [
            row["audio_filepath"]
            for row in rows
            if not Path(row["audio_filepath"]).exists()
        ]

        validation[view_name] = {
            "row_count": len(rows),
            "positions_ordered":
            positions == list(range(len(rows))),
            "segment_ids_unique":
            len(ids) == len(set(ids)),
            "missing_audio_count":
            len(missing_audio),
        }

        validation[view_name]["passed"] = (
            validation[view_name]["positions_ordered"]
            and validation[view_name]["segment_ids_unique"]
            and not missing_audio
        )

    expected_canonical = math.ceil(duration / 20.0)

    passed = (
        all(item["passed"] for item in validation.values())
        and len(view_rows["canonical_20s"])
        == expected_canonical
    )

    report = {
        "schema_version":
        "silver_v3_multiview_preparation_report_v1",
        "lecture_id":
        args.lecture_id,
        "source_audio":
        str(args.audio),
        "source_sha256":
        sha256_file(args.audio),
        "sample_rate":
        sample_rate,
        "duration_seconds":
        round(duration, 6),
        "wall_seconds":
        round(time.perf_counter() - started, 3),
        "view_counts": {
            view: len(rows)
            for view, rows in view_rows.items()
        },
        "total_inference_items":
        sum(len(rows) for rows in view_rows.values()),
        "canonical_manifest":
        str(canonical_manifest),
        "validation":
        validation,
        "passed":
        passed,
    }

    report_path = (
        args.output_root
        / "silver_v3_multiview_preparation_report.json"
    )

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

    print(json.dumps(report, ensure_ascii=False, indent=2))

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
