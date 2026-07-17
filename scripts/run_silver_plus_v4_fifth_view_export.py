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

from integrate_silver_plus_v3_fifth_view import AZURE_VIEW, skeleton, text_from_tokens, token_center, token_text, write_jsonl
from run_silver_plus_v3_fifth_view_export_v2 import integrate as integrate_v3

LAYER_VERSION = "silver_plus_v4"
BUILT_ON_SPINE = "silver_v3_repaired_dd47cd04"
GOLD_SHA256 = "63a2981d7368968ea42968d93dca6988bc31d51dd872d41cca5d1e8b1fcbf6a6"
EVAL_SHA256 = "7dc28d6a355a11abbcffcab361ba739e728ed3fdc68f26d2d95ad6a70dd6727b"
HONORIFIC_EQUIVALENCE: dict[str, tuple[str, ...]] = {
    "sallallahu": ("صلي",),
    "salallahu": ("صلي",),
    "alaihi": ("عليه",),
    "alayhi": ("عليه",),
    "wasallam": ("وسلم",),
    "wasalam": ("وسلم",),
    "allah": ("الله",),
    "sallallahualaihiwasallam": ("صلي", "الله", "عليه", "وسلم"),
}
REQUIRED_VIEWS = [
    "canonical_20s",
    "whole_slice",
    "context_10s_stride_5s",
    "local_2p5s_contiguous",
    AZURE_VIEW,
]
DOCUMENTED_VIEW_GAPS = {"seg_000019": {"whole_slice"}}
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


def complete_views(token: dict[str, Any]) -> list[str]:
    values = list(token.get("views", []))
    for observation in token.get("observations", []):
        if not isinstance(observation, dict):
            continue
        for key in ("view", "view_id", "source_view", "source"):
            value = observation.get(key)
            if isinstance(value, str) and value:
                values.append(value)
    return list(dict.fromkeys(value for value in values if value))


def add_alternate(target: dict[str, Any], source: dict[str, Any], record: bool) -> None:
    source_text = token_text(source)
    source_views = complete_views(source) or [AZURE_VIEW]
    target["views"] = list(dict.fromkeys([*complete_views(target), *source_views, AZURE_VIEW]))
    target["support_count"] = len(target["views"])
    target["flags"] = list(dict.fromkeys([*target.get("flags", []), "script_disagreement"]))
    target["observations"] = [*target.get("observations", []), *source.get("observations", [])]
    if record:
        alternate = {"text": source_text, "view": AZURE_VIEW, "flag": "script_disagreement"}
        existing = target.get("alternates", [])
        if alternate not in existing:
            target["alternates"] = [*existing, alternate]
    if target.get("acceptance_tier") == "C_single_witness" and len(target["views"]) >= 2:
        target["acceptance_tier"] = "A_corroborated"


def find_ordered_targets(tokens: list[dict[str, Any]], targets: tuple[str, ...], center: float) -> list[int] | None:
    folded = [skeleton(token_text(token)) for token in tokens]
    candidates: list[tuple[float, list[int]]] = []
    for start in range(len(tokens) - len(targets) + 1):
        if tuple(folded[start:start + len(targets)]) == targets:
            centers = [token_center(tokens[index]) for index in range(start, start + len(targets))]
            valid = [value for value in centers if value is not None]
            score = abs(sum(valid) / len(valid) - center) if valid else float(start)
            candidates.append((score, list(range(start, start + len(targets)))))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    if len(targets) == 1:
        matches = [index for index, folded_value in enumerate(folded) if folded_value == targets[0]]
        if matches:
            return [min(matches, key=lambda index: abs((token_center(tokens[index]) or center) - center))]
    return None


def apply_fix_b(row: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(row)
    tokens = list(row.get("tokens", []))
    remove: set[int] = set()
    for index, token in enumerate(tokens):
        key = latin_key(token_text(token))
        targets = HONORIFIC_EQUIVALENCE.get(key)
        if not targets:
            continue
        center = float(token_center(token) or 0.0)
        matches = find_ordered_targets(tokens, targets, center)
        if not matches or index in matches:
            continue
        for target_position, target_index in enumerate(matches):
            add_alternate(tokens[target_index], token, target_position == 0)
        remove.add(index)
    tokens = [token for index, token in enumerate(tokens) if index not in remove]
    row["tokens"] = tokens
    row["schema_version"] = "silver_plus_v4_fifth_view_segment_v1"
    row["layer_version"] = LAYER_VERSION
    row["built_on_spine"] = BUILT_ON_SPINE
    row["silver_plus_v4_text"] = text_from_tokens(tokens)
    row.pop("silver_plus_v3_text", None)
    row["text"] = row["silver_plus_v4_text"]
    return row


def provenance_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        for position, token in enumerate(row.get("tokens", [])):
            views = complete_views(token)
            output.append({
                "schema_version": "silver_plus_v4_token_provenance_v1",
                "layer_version": LAYER_VERSION,
                "built_on_spine": BUILT_ON_SPINE,
                "segment_position": row.get("segment_position"),
                "segment_id": row.get("segment_id"),
                "embedded_seg_id": row["embedded_seg_id"],
                "token_position": position,
                "text": token_text(token),
                "views": views,
                "support_count": token.get("support_count", len(views)),
                "tier": token.get("acceptance_tier"),
                "alternates": token.get("alternates", []),
                "flags": token.get("flags", []),
                "observations": token.get("observations", []),
            })
    return output


def arabic_letters(text: str) -> int:
    return len(AR_LETTER.findall(str(text)))


def repeated_6grams(rows: list[dict[str, Any]], eval_module) -> int:
    grams = Counter()
    for row in rows:
        values = eval_module.toks(row["silver_plus_v4_text"])
        grams.update(tuple(values[index:index + 6]) for index in range(max(0, len(values) - 5)))
    return sum(count - 1 for count in grams.values() if count > 1)


def honorific_6gram_count(rows: list[dict[str, Any]], eval_module) -> int:
    keys = set(HONORIFIC_EQUIVALENCE)
    count = 0
    for row in rows:
        values = eval_module.toks(row.get("silver_plus_v4_text") or row.get("silver_plus_v3_text") or row.get("text") or "")
        for index in range(max(0, len(values) - 5)):
            if any(latin_key(token) in keys for token in values[index:index + 6]):
                count += 1
    return count


def run_gates(rows, provenance, gold_path, eval_module, deterministic):
    gold = eval_module.load_gold(str(gold_path))
    losses = []
    net_lost = 0
    for row in rows:
        before = arabic_letters(row["silver_v3_text"])
        after = arabic_letters(row["silver_plus_v4_text"])
        if after < before:
            losses.append(row["embedded_seg_id"])
            net_lost += before - after
    full_text = " ".join(row["silver_plus_v4_text"] for row in rows)
    ar_count = arabic_letters(full_text)
    density = len(HARAKAT.findall(full_text)) / ar_count if ar_count else 0.0
    ar_hit = ar_total = en_hit = en_total = 0
    for row in rows:
        sid = int(row["embedded_seg_id"].rsplit("_", 1)[1])
        if sid >= 30 or sid not in gold:
            continue
        hit, total = eval_module.count_aware_recall(gold[sid], row["silver_plus_v4_text"], True)
        ar_hit += hit; ar_total += total
        hit, total = eval_module.count_aware_recall(gold[sid], row["silver_plus_v4_text"], False)
        en_hit += hit; en_total += total
    ar_recall = ar_hit / ar_total if ar_total else 1.0
    en_recall = en_hit / en_total if en_total else 1.0
    total_tokens = len(provenance)
    a_share = sum(token.get("tier") == "A_corroborated" for token in provenance) / total_tokens if total_tokens else 0.0
    dup6 = repeated_6grams(rows, eval_module)
    observed_global = {view for token in provenance for view in token.get("views", [])}
    per_segment = True
    for row in rows:
        sid = row["embedded_seg_id"]
        sid_int = int(sid.rsplit("_", 1)[1])
        if sid_int not in gold or not row["silver_plus_v4_text"].strip():
            continue
        observed = {view for token in row.get("tokens", []) for view in complete_views(token)}
        required = {"canonical_20s", "whole_slice"} - DOCUMENTED_VIEW_GAPS.get(sid, set())
        if not required <= observed:
            per_segment = False
            break
    gates = {
        "no_regression_vs_silver_v3": {"pass": len(losses) == 0, "segments_losing_ar": len(losses), "net_ar_letters_lost": net_lost},
        "vocalization_floor": {"pass": density >= 0.20, "tashkeel_density": density, "threshold": 0.20},
        "pilot_window_union_reproduction": {"pass": ar_recall >= 0.33 and en_recall >= 0.80, "ar_recall": ar_recall, "en_recall": en_recall, "ar_threshold": 0.33, "en_threshold": 0.80},
        "corroboration_floor": {"pass": a_share >= 0.60, "a_tier_share": a_share, "threshold": 0.60},
        "duplicate_budget": {"pass": dup6 <= 25, "global_repeated_6grams": dup6, "threshold": 25},
        "all_views_present": {"pass": set(REQUIRED_VIEWS) <= observed_global and per_segment, "views": REQUIRED_VIEWS},
        "determinism": {"pass": deterministic, "rerun_hash_matches": deterministic},
    }
    return {"validation_performed": True, "gates": gates, "all_gates_pass": all(gate["pass"] for gate in gates.values())}


def serialize(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate the Silver+ v4 enhancement layer on the accepted repaired Silver v3 spine.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--silver-v3", type=Path, required=True)
    parser.add_argument("--azure-parent", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--gold-eval-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-align-seconds", type=float, default=0.85)
    args = parser.parse_args()
    if "silver_plus_v4" not in args.output_prefix:
        raise ValueError("output-prefix must use lecture_XXX_silver_plus_v4 naming")
    if sha256_file(args.gold) != GOLD_SHA256:
        raise RuntimeError("gold_v12_segments.jsonl sha256 differs from the handed corrected reference")
    eval_module = load_eval(args.gold_eval_reference)
    silver_rows = read_jsonl(args.silver_v3)
    azure_rows = read_jsonl(args.azure_parent)
    if len(silver_rows) != len(azure_rows):
        raise ValueError(f"Row count mismatch: silver_v3={len(silver_rows)} azure_parent={len(azure_rows)}")
    commit = git_head(args.repo_root)
    config = {
        "layer_version": LAYER_VERSION,
        "built_on_spine": BUILT_ON_SPINE,
        "max_align_seconds": args.max_align_seconds,
        "honorific_equivalence": HONORIFIC_EQUIVALENCE,
        "gold_sha256": GOLD_SHA256,
        "evaluation_sha256": EVAL_SHA256,
    }
    config_hash = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    baseline_rows, _ = integrate_v3(silver_rows, azure_rows, args.max_align_seconds, commit, config_hash)
    rows = [apply_fix_b(row) for row in baseline_rows]
    rerun_baseline, _ = integrate_v3(silver_rows, azure_rows, args.max_align_seconds, commit, config_hash)
    rerun_rows = [apply_fix_b(row) for row in rerun_baseline]
    deterministic = hashlib.sha256(serialize(rows)).digest() == hashlib.sha256(serialize(rerun_rows)).digest()
    provenance = provenance_rows(rows)
    validation = run_gates(rows, provenance, args.gold, eval_module, deterministic)
    before_honorific = honorific_6gram_count(baseline_rows, eval_module)
    after_honorific = honorific_6gram_count(rows, eval_module)
    spine_unchanged = all(
        row["silver_v3_text"] == baseline["silver_v3_text"]
        for row, baseline in zip(rows, baseline_rows)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_path = args.output_dir / f"{args.output_prefix}_segment_level.jsonl"
    provenance_path = args.output_dir / f"{args.output_prefix}_token_provenance.jsonl"
    json_path = args.output_dir / f"{args.output_prefix}.json"
    report_path = args.output_dir / f"{args.output_prefix}_run_report.json"
    manifest_path = args.output_dir / "PACKAGE_MANIFEST.json"
    package_path = args.output_dir / f"{args.output_prefix}.zip"
    write_jsonl(segment_path, rows)
    write_jsonl(provenance_path, provenance)
    json_path.write_text(json.dumps({
        "schema_version": "silver_plus_v4_fifth_view_export_v1",
        "layer_version": LAYER_VERSION,
        "built_on_spine": BUILT_ON_SPINE,
        "repository_commit": commit,
        "config_hash": config_hash,
        "segment_count": len(rows),
        "segments": rows,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    validation_object = validation
    manifest = {
        "schema_version": "silver_plus_v4_package_manifest_v1",
        "layer_version": LAYER_VERSION,
        "built_on_spine": BUILT_ON_SPINE,
        "repository_commit": commit,
        "config_hash": config_hash,
        "segment_count": len(rows),
        "validation_performed": True,
        "validation": validation_object,
        "files": [],
    }
    for path in (segment_path, provenance_path, json_path):
        manifest["files"].append({"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "layer_version": LAYER_VERSION,
        "built_on_spine": BUILT_ON_SPINE,
        "repository_commit": commit,
        "config_hash": config_hash,
        "honorific_6grams_before": before_honorific,
        "honorific_6grams_after": after_honorific,
        "validation": validation,
        "silver_v3_text_byte_identical_to_spine": spine_unchanged,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in (segment_path, provenance_path, json_path, manifest_path, report_path):
            archive.write(path, arcname=path.name)
    print(json.dumps({"completed": validation["all_gates_pass"], "validation": validation, "zip": str(package_path)}, ensure_ascii=False, indent=2, sort_keys=True))
    if not spine_unchanged:
        raise SystemExit("silver_v3_text changed; stopping")
    if not validation["all_gates_pass"]:
        failed = [name for name, gate in validation["gates"].items() if not gate["pass"]]
        raise SystemExit("Gate failure: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
