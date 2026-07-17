from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import subprocess
import zipfile
from collections import Counter
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
    skeleton,
    text_from_tokens,
    token_center,
    token_text,
    write_jsonl,
)

GOLD_SHA256 = "e27d0dbab878cdeef0ba70f9f935969c51476e61d4d0f750769bfb1aebe9a094"
EVAL_SHA256 = "7dc28d6a355a11abbcffcab361ba739e728ed3fdc68f26d2d95ad6a70dd6727b"
HONORIFIC_EQUIVALENCE = {
    "sallallahu": ("صلي",),
    "alaihi": ("عليه",),
    "wasallam": ("وسلم",),
    "allah": ("الله",),
    "salallahu": ("صلي",),
    "alayhi": ("عليه",),
    "wasalam": ("وسلم",),
    "sallallahualaihiwasallam": ("صلي", "الله", "عليه", "وسلم"),
}
AR_LETTER = re.compile(r"[\u0621-\u064A]")
HARAKAT = re.compile(r"[\u064B-\u0652\u0670]")
LATIN_CLEAN = re.compile(r"[^a-z]")


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
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()


def load_eval(path: Path):
    if sha256_file(path) != EVAL_SHA256:
        raise RuntimeError("gold_eval_reference.py sha256 differs from the handed reference")
    spec = importlib.util.spec_from_file_location("gold_eval_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load gold_eval_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latin_key(text: str) -> str:
    return LATIN_CLEAN.sub("", str(text).lower())


def expand_honorific_observation(obs: dict[str, Any]) -> list[dict[str, Any]]:
    key = latin_key(obs["text"])
    targets = HONORIFIC_EQUIVALENCE.get(key)
    if not targets:
        return [obs]
    start = float(obs["global_start"])
    end = float(obs["global_end"])
    width = max(0.001, end - start)
    expanded = []
    for index, target in enumerate(targets):
        item = dict(obs)
        item["global_start"] = round(start + width * index / len(targets), 6)
        item["global_end"] = round(start + width * (index + 1) / len(targets), 6)
        item["center"] = round((item["global_start"] + item["global_end"]) / 2, 6)
        item["honorific_target_skeleton"] = target
        item["honorific_original"] = obs["text"]
        item["honorific_record_alternate"] = index == 0
        expanded.append(item)
    return expanded


def honorific_match(tokens: list[dict[str, Any]], obs: dict[str, Any], max_seconds: float) -> int | None:
    target = obs.get("honorific_target_skeleton")
    if not target:
        return None
    center = float(obs["center"])
    candidates = []
    for index, token in enumerate(tokens):
        token_mid = token_center(token)
        if token_mid is None or abs(token_mid - center) > max_seconds:
            continue
        if skeleton(token_text(token)) == target:
            candidates.append((abs(token_mid - center), index))
    return min(candidates)[1] if candidates else None


def corroborate_honorific(token: dict[str, Any], obs: dict[str, Any]) -> None:
    views = list(dict.fromkeys([*token.get("views", []), AZURE_VIEW]))
    token["views"] = views
    token["support_count"] = len(views)
    token["observations"] = [*token.get("observations", []), obs]
    token["flags"] = list(dict.fromkeys([*token.get("flags", []), "script_disagreement"]))
    if obs.get("honorific_record_alternate", True):
        alternate = {
            "text": obs["honorific_original"],
            "view": AZURE_VIEW,
            "flag": "script_disagreement",
        }
        token["alternates"] = [*token.get("alternates", []), alternate]
    if token.get("acceptance_tier") == "C_single_witness" and len(views) >= 2:
        token["acceptance_tier"] = "A_corroborated"


def integrate(
    silver_rows: list[dict[str, Any]],
    azure_rows: list[dict[str, Any]],
    max_align_seconds: float,
    commit: str,
    config_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for position, (silver, azure) in enumerate(zip(silver_rows, azure_rows)):
        row = copy.deepcopy(silver)
        tokens = list(row.get("tokens", []))
        seg_start, seg_end = segment_span(row)
        silver_v3_text = text_from_tokens(tokens) or str(row.get("text") or "")
        observations = []
        for raw in azure_observations(azure, seg_start, seg_end):
            observations.extend(expand_honorific_observation(raw))
        for observation in observations:
            if observation.get("honorific_target_skeleton"):
                match = honorific_match(tokens, observation, max_align_seconds)
                if match is not None:
                    corroborate_honorific(tokens[match], observation)
                    continue
            match = choose_match(tokens, observation, max_align_seconds)
            if match is None:
                tokens.append(azure_only_token(observation))
            else:
                corroborate(tokens[match], observation)
        tokens.sort(key=lambda token: (float(token_center(token) or seg_start), str(token.get("word_position", ""))))
        integrated_text = text_from_tokens(tokens)
        row.update({
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
        })
        output_rows.append(row)
        for token_position, token in enumerate(tokens):
            provenance_rows.append({
                "schema_version": "silver_plus_v3_token_provenance_v1",
                "segment_position": position,
                "segment_id": row.get("segment_id"),
                "embedded_seg_id": seg_id(position),
                "token_position": token_position,
                "text": token_text(token),
                "views": token.get("views", []),
                "support_count": token.get("support_count", len(token.get("views", []))),
                "tier": token.get("acceptance_tier"),
                "alternates": token.get("alternates", []),
                "flags": token.get("flags", []),
                "observations": token.get("observations", []),
                "is_arabic": is_arabic(token_text(token)),
            })
    return output_rows, provenance_rows


def arabic_letters(text: str) -> int:
    return len(AR_LETTER.findall(str(text)))


def repeated_6grams(rows: list[dict[str, Any]], eval_module) -> int:
    grams = Counter()
    for row in rows:
        values = eval_module.toks(row["silver_plus_v3_text"])
        grams.update(tuple(values[i:i + 6]) for i in range(max(0, len(values) - 5)))
    return sum(count - 1 for count in grams.values() if count > 1)


def run_gates(output_rows, provenance_rows, gold_path, eval_module, rerun_hash_matches):
    gold = eval_module.load_gold(str(gold_path))
    losses = []
    net_lost = 0
    for row in output_rows:
        before = arabic_letters(row["silver_v3_text"])
        after = arabic_letters(row["silver_plus_v3_text"])
        if after < before:
            losses.append(row["embedded_seg_id"])
            net_lost += before - after
    full_text = " ".join(row["silver_plus_v3_text"] for row in output_rows)
    ar_count = arabic_letters(full_text)
    density = len(HARAKAT.findall(full_text)) / ar_count if ar_count else 0.0
    ar_hit = ar_total = en_hit = en_total = 0
    for row in output_rows:
        sid = int(row["embedded_seg_id"].rsplit("_", 1)[1])
        if sid >= 30 or sid not in gold:
            continue
        hit, total = eval_module.count_aware_recall(gold[sid], row["silver_plus_v3_text"], True)
        ar_hit += hit; ar_total += total
        hit, total = eval_module.count_aware_recall(gold[sid], row["silver_plus_v3_text"], False)
        en_hit += hit; en_total += total
    ar_recall = ar_hit / ar_total if ar_total else 1.0
    en_recall = en_hit / en_total if en_total else 1.0
    total_tokens = len(provenance_rows)
    a_share = sum(row.get("tier") == "A_corroborated" for row in provenance_rows) / total_tokens if total_tokens else 0.0
    dup6 = repeated_6grams(output_rows, eval_module)
    observed = {view for row in provenance_rows for view in row.get("views", [])}
    normalized = {"local_2p5s" if view == "local_2p5s_contiguous" else view for view in observed}
    required = ["canonical_20s", "whole_slice", "context_10s_stride_5s", "local_2p5s", AZURE_VIEW]
    per_segment = all(
        {"canonical_20s", "whole_slice"} <= {"local_2p5s" if v == "local_2p5s_contiguous" else v for token in row.get("tokens", []) for v in token.get("views", [])}
        for row in output_rows if row["silver_plus_v3_text"].strip()
    )
    gates = {
        "no_regression_vs_silver_v3": {"pass": len(losses) == 0, "segments_losing_ar": len(losses), "net_ar_letters_lost": net_lost},
        "vocalization_floor": {"pass": density >= 0.20, "tashkeel_density": density, "threshold": 0.20},
        "pilot_window_union_reproduction": {"pass": ar_recall >= 0.33 and en_recall >= 0.80, "ar_recall": ar_recall, "en_recall": en_recall, "ar_threshold": 0.33, "en_threshold": 0.80},
        "corroboration_floor": {"pass": a_share >= 0.60, "a_tier_share": a_share, "threshold": 0.60},
        "duplicate_budget": {"pass": dup6 <= 25, "global_repeated_6grams": dup6, "threshold": 25},
        "all_views_present": {"pass": set(required) <= normalized and per_segment, "views": required},
        "determinism": {"pass": rerun_hash_matches, "rerun_hash_matches": rerun_hash_matches},
    }
    return {"validation_performed": True, "gates": gates, "all_gates_pass": all(gate["pass"] for gate in gates.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run accepted Silver+ v3 fifth-view integration with exact embedded gates.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--silver-v3", type=Path, required=True)
    parser.add_argument("--azure-parent", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--gold-eval-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-align-seconds", type=float, default=0.85)
    args = parser.parse_args()
    if sha256_file(args.gold) != GOLD_SHA256:
        raise RuntimeError("gold_v12_segments.jsonl sha256 differs from the handed reference")
    eval_module = load_eval(args.gold_eval_reference)
    silver_rows = read_jsonl(args.silver_v3)
    azure_rows = read_jsonl(args.azure_parent)
    if len(silver_rows) != len(azure_rows):
        raise ValueError(f"Row count mismatch: silver_v3={len(silver_rows)} azure_parent={len(azure_rows)}")
    commit = git_head(args.repo_root)
    config = {"max_align_seconds": args.max_align_seconds, "honorific_equivalence": HONORIFIC_EQUIVALENCE}
    config_hash = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_rows, provenance_rows = integrate(silver_rows, azure_rows, args.max_align_seconds, commit, config_hash)
    rerun_rows, _ = integrate(silver_rows, azure_rows, args.max_align_seconds, commit, config_hash)
    serialized = lambda rows: "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()
    validation = run_gates(output_rows, provenance_rows, args.gold, eval_module, hashlib.sha256(serialized(output_rows)).digest() == hashlib.sha256(serialized(rerun_rows)).digest())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_path = args.output_dir / f"{args.output_prefix}_segment_level.jsonl"
    provenance_path = args.output_dir / f"{args.output_prefix}_token_provenance.jsonl"
    json_path = args.output_dir / f"{args.output_prefix}.json"
    report_path = args.output_dir / f"{args.output_prefix}_run_report.json"
    manifest_path = args.output_dir / "PACKAGE_MANIFEST.json"
    package_path = args.output_dir / f"{args.output_prefix}.zip"
    write_jsonl(segment_path, output_rows)
    write_jsonl(provenance_path, provenance_rows)
    json_path.write_text(json.dumps({"schema_version":"silver_plus_v3_fifth_view_export_v1","repository_commit":commit,"config_hash":config_hash,"segment_count":len(output_rows),"segments":output_rows}, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest = {"schema_version":"silver_plus_v3_package_manifest_v1","repository_commit":commit,"config_hash":config_hash,"segment_count":len(output_rows),"validation_performed":True,"validation":validation,"files":[]}
    for path in (segment_path, provenance_path, json_path):
        manifest["files"].append({"filename":path.name,"size_bytes":path.stat().st_size,"sha256":sha256_file(path)})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps({"repository_commit":commit,"config_hash":config_hash,"validation":validation,"silver_v3_text_byte_unchanged":True}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in (segment_path, provenance_path, json_path, manifest_path, report_path):
            archive.write(path, arcname=path.name)
    print(json.dumps({"completed":validation["all_gates_pass"],"validation":validation,"zip":str(package_path)}, ensure_ascii=False, indent=2, sort_keys=True))
    if not validation["all_gates_pass"]:
        failed = [name for name, gate in validation["gates"].items() if not gate["pass"]]
        raise SystemExit("Gate failure: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
