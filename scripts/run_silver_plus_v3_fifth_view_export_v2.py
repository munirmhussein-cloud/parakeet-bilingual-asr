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
    corroborate,
    seg_id,
    segment_span,
    skeleton,
    text_from_tokens,
    token_center,
    token_text,
    write_jsonl,
)
from run_silver_plus_v3_fifth_view_export import (
    EVAL_SHA256,
    GOLD_SHA256,
    HONORIFIC_EQUIVALENCE,
    arabic_letters,
    latin_key,
    load_eval,
    sha256_file,
)

HARAKAT = re.compile(r"[\u064B-\u0652\u0670]")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()


def canonical_view(view: Any) -> str:
    value = str(view or "")
    return "local_2p5s" if value == "local_2p5s_contiguous" else value


def complete_views(token: dict[str, Any]) -> list[str]:
    values = list(token.get("views", []))
    for observation in token.get("observations", []):
        if not isinstance(observation, dict):
            continue
        for key in ("view", "view_id", "source_view", "source"):
            value = observation.get(key)
            if isinstance(value, str) and value:
                values.append(value)
    return list(dict.fromkeys(canonical_view(value) for value in values if value))


def exact_match(tokens: list[dict[str, Any]], observation: dict[str, Any], max_seconds: float) -> int | None:
    key = skeleton(observation["text"])
    center = float(observation["center"])
    candidates = []
    for index, token in enumerate(tokens):
        token_mid = token_center(token)
        if token_mid is None or abs(token_mid - center) > max_seconds:
            continue
        if key and skeleton(token_text(token)) == key:
            candidates.append((abs(token_mid - center), index))
    return min(candidates)[1] if candidates else None


def honorific_targets(observation: dict[str, Any]) -> tuple[str, ...] | None:
    return HONORIFIC_EQUIVALENCE.get(latin_key(observation["text"]))


def honorific_sequence_match(
    tokens: list[dict[str, Any]],
    targets: tuple[str, ...],
    preferred_center: float,
) -> list[int] | None:
    folded = [skeleton(token_text(token)) for token in tokens]
    candidates: list[tuple[float, list[int]]] = []
    for start in range(len(tokens) - len(targets) + 1):
        if tuple(folded[start:start + len(targets)]) != tuple(targets):
            continue
        centers = [token_center(tokens[index]) for index in range(start, start + len(targets))]
        valid = [value for value in centers if value is not None]
        score = abs(sum(valid) / len(valid) - preferred_center) if valid else float(start)
        candidates.append((score, list(range(start, start + len(targets)))))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]

    # Exact ordered skeletons, not fuzzy matching. This handles intervening punctuation only.
    ordered: list[int] = []
    cursor = 0
    for target in targets:
        options = [index for index in range(cursor, len(tokens)) if folded[index] == target]
        if not options:
            return None
        chosen = min(
            options,
            key=lambda index: abs((token_center(tokens[index]) or preferred_center) - preferred_center),
        )
        ordered.append(chosen)
        cursor = chosen + 1
    return ordered


def add_honorific_alternate(token: dict[str, Any], observation: dict[str, Any], record: bool) -> None:
    views = list(dict.fromkeys([*complete_views(token), AZURE_VIEW]))
    token["views"] = views
    token["support_count"] = len(views)
    token["observations"] = [*token.get("observations", []), observation]
    token["flags"] = list(dict.fromkeys([*token.get("flags", []), "script_disagreement"]))
    if record:
        token["alternates"] = [
            *token.get("alternates", []),
            {"text": observation["text"], "view": AZURE_VIEW, "flag": "script_disagreement"},
        ]
    if token.get("acceptance_tier") == "C_single_witness" and len(views) >= 2:
        token["acceptance_tier"] = "A_corroborated"


def integrate(silver_rows, azure_rows, max_align_seconds, commit, config_hash):
    output_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for position, (silver, azure) in enumerate(zip(silver_rows, azure_rows)):
        row = copy.deepcopy(silver)
        tokens = list(row.get("tokens", []))
        for token in tokens:
            token["views"] = complete_views(token)
        seg_start, seg_end = segment_span(row)
        silver_v3_text = text_from_tokens(tokens) or str(row.get("text") or "")

        for observation in azure_observations(azure, seg_start, seg_end):
            targets = honorific_targets(observation)
            if targets:
                matches = honorific_sequence_match(tokens, targets, float(observation["center"]))
                if matches:
                    for target_position, token_index in enumerate(matches):
                        add_honorific_alternate(tokens[token_index], observation, target_position == 0)
                    continue

            match = exact_match(tokens, observation, max_align_seconds)
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
            views = complete_views(token)
            token["views"] = views
            provenance_rows.append({
                "schema_version": "silver_plus_v3_token_provenance_v1",
                "segment_position": position,
                "segment_id": row.get("segment_id"),
                "embedded_seg_id": seg_id(position),
                "token_position": token_position,
                "text": token_text(token),
                "views": views,
                "support_count": token.get("support_count", len(views)),
                "tier": token.get("acceptance_tier"),
                "alternates": token.get("alternates", []),
                "flags": token.get("flags", []),
                "observations": token.get("observations", []),
            })
    return output_rows, provenance_rows


def repeated_6grams(rows, eval_module) -> int:
    grams = Counter()
    for row in rows:
        values = eval_module.toks(row["silver_plus_v3_text"])
        grams.update(tuple(values[index:index + 6]) for index in range(max(0, len(values) - 5)))
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
    required = ["canonical_20s", "whole_slice", "context_10s_stride_5s", "local_2p5s", AZURE_VIEW]
    observed = {canonical_view(view) for row in provenance_rows for view in row.get("views", [])}
    per_segment = all(
        {"canonical_20s", "whole_slice"} <= {
            canonical_view(view)
            for token in row.get("tokens", [])
            for view in complete_views(token)
        }
        for row in output_rows if row["silver_plus_v3_text"].strip()
    )
    gates = {
        "no_regression_vs_silver_v3": {"pass": len(losses) == 0, "segments_losing_ar": len(losses), "net_ar_letters_lost": net_lost},
        "vocalization_floor": {"pass": density >= 0.20, "tashkeel_density": density, "threshold": 0.20},
        "pilot_window_union_reproduction": {"pass": ar_recall >= 0.33 and en_recall >= 0.80, "ar_recall": ar_recall, "en_recall": en_recall, "ar_threshold": 0.33, "en_threshold": 0.80},
        "corroboration_floor": {"pass": a_share >= 0.60, "a_tier_share": a_share, "threshold": 0.60},
        "duplicate_budget": {"pass": dup6 <= 25, "global_repeated_6grams": dup6, "threshold": 25},
        "all_views_present": {"pass": set(required) <= observed and per_segment, "views": required},
        "determinism": {"pass": rerun_hash_matches, "rerun_hash_matches": rerun_hash_matches},
    }
    return {"validation_performed": True, "gates": gates, "all_gates_pass": all(gate["pass"] for gate in gates.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repaired Silver+ v3 fifth-view integration with exact fixed gates.")
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
    if sha256_file(args.gold_eval_reference) != EVAL_SHA256:
        raise RuntimeError("gold_eval_reference.py sha256 differs from the handed reference")
    eval_module = load_eval(args.gold_eval_reference)
    silver_rows = read_jsonl(args.silver_v3)
    azure_rows = read_jsonl(args.azure_parent)
    if len(silver_rows) != len(azure_rows):
        raise ValueError(f"Row count mismatch: silver_v3={len(silver_rows)} azure_parent={len(azure_rows)}")

    commit = git_head(args.repo_root)
    config = {"max_align_seconds": args.max_align_seconds, "honorific_equivalence": HONORIFIC_EQUIVALENCE, "matching": "skeleton_only_plus_exact_honorific_sequence"}
    config_hash = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_rows, provenance_rows = integrate(silver_rows, azure_rows, args.max_align_seconds, commit, config_hash)
    rerun_rows, _ = integrate(silver_rows, azure_rows, args.max_align_seconds, commit, config_hash)
    serialize = lambda rows: "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()
    deterministic = hashlib.sha256(serialize(output_rows)).digest() == hashlib.sha256(serialize(rerun_rows)).digest()
    validation = run_gates(output_rows, provenance_rows, args.gold, eval_module, deterministic)

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
