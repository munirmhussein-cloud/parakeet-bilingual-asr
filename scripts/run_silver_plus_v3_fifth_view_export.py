from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from integrate_silver_plus_v3_fifth_view import (
    AZURE_VIEW,
    azure_observations,
    azure_only_token,
    choose_match,
    corroborate,
    is_arabic,
    seg_id,
    segment_span,
    text_from_tokens,
    token_center,
    token_text,
    write_jsonl,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Silver+ v3 fifth-view integration and package export without Gold/reference validation."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--silver-v3", type=Path, required=True)
    parser.add_argument("--azure-parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-align-seconds", type=float, default=0.85)
    args = parser.parse_args()

    silver_rows = read_jsonl(args.silver_v3)
    azure_rows = read_jsonl(args.azure_parent)
    if len(silver_rows) != len(azure_rows):
        raise ValueError(
            f"Row count mismatch: silver_v3={len(silver_rows)} azure_parent={len(azure_rows)}"
        )

    commit = git_head(args.repo_root)
    config = {
        "max_align_seconds": args.max_align_seconds,
        "view_priority": [
            "canonical_20s",
            "whole_slice",
            "context_10s_stride_5s",
            "local_2p5s_contiguous",
            AZURE_VIEW,
        ],
        "azure_vote_key": "tashkeel_stripped_light_orthographic_fold",
        "validation_mode": "none",
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    output_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []

    for position, (silver, azure) in enumerate(zip(silver_rows, azure_rows)):
        row = copy.deepcopy(silver)
        tokens = list(row.get("tokens", []))
        seg_start, seg_end = segment_span(row)
        silver_v3_text = text_from_tokens(tokens) or str(row.get("text") or "")

        for observation in azure_observations(azure, seg_start, seg_end):
            match = choose_match(tokens, observation, args.max_align_seconds)
            if match is None:
                tokens.append(azure_only_token(observation))
            else:
                corroborate(tokens[match], observation)

        tokens.sort(
            key=lambda token: (
                float(token_center(token) or seg_start),
                str(token.get("word_position", "")),
            )
        )
        integrated_text = text_from_tokens(tokens)

        row.update(
            {
                "schema_version": "silver_plus_v3_fifth_view_segment_v1",
                "segment_position": position,
                "segment_index": position,
                "embedded_seg_id": seg_id(position),
                "silver_v3_text": silver_v3_text,
                "silver_plus_v3_text": integrated_text,
                "text": integrated_text,
                "tokens": tokens,
                "reconciliation": {
                    "engine_spine": "repaired_silver_v3_multiview_lattice",
                    "azure_used_inside_silver_v3_lattice": True,
                    "azure_view_priority": "lowest",
                    "azure_votes_on_skeleton_only": True,
                    "vocalized_parakeet_surface_preferred": True,
                    "parent_level_resolution_only": False,
                    "token_level_merge_used": True,
                    "repository_commit": commit,
                    "config_hash": config_hash,
                },
            }
        )
        output_rows.append(row)

        for token_position, token in enumerate(tokens):
            provenance_rows.append(
                {
                    "schema_version": "silver_plus_v3_token_provenance_v1",
                    "segment_position": position,
                    "segment_id": row.get("segment_id"),
                    "embedded_seg_id": seg_id(position),
                    "token_position": token_position,
                    "text": token_text(token),
                    "views": token.get("views", []),
                    "support_count": token.get(
                        "support_count", len(token.get("views", []))
                    ),
                    "tier": token.get("acceptance_tier"),
                    "alternates": token.get("alternates", []),
                    "flags": token.get("flags", []),
                    "observations": token.get("observations", []),
                    "is_arabic": is_arabic(token_text(token)),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_path = args.output_dir / f"{args.output_prefix}_segment_level.jsonl"
    provenance_path = args.output_dir / f"{args.output_prefix}_token_provenance.jsonl"
    json_path = args.output_dir / f"{args.output_prefix}.json"
    package_path = args.output_dir / f"{args.output_prefix}.zip"

    write_jsonl(segment_path, output_rows)
    write_jsonl(provenance_path, provenance_rows)
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "silver_plus_v3_fifth_view_export_v1",
                "repository_commit": commit,
                "config_hash": config_hash,
                "segment_count": len(output_rows),
                "segments": output_rows,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "silver_plus_v3_package_manifest_v1",
        "repository_commit": commit,
        "config_hash": config_hash,
        "segment_count": len(output_rows),
        "validation_performed": False,
        "files": [],
    }

    for path in (segment_path, provenance_path, json_path):
        manifest["files"].append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    with zipfile.ZipFile(
        package_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in (segment_path, provenance_path, json_path):
            archive.write(path, arcname=path.name)
        archive.writestr(
            "PACKAGE_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    print(
        json.dumps(
            {
                "completed": True,
                "validation_performed": False,
                "repository_commit": commit,
                "segment_count": len(output_rows),
                "segment_jsonl": str(segment_path),
                "token_provenance_jsonl": str(provenance_path),
                "json": str(json_path),
                "zip": str(package_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
