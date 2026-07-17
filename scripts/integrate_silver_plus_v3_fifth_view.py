from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

AZURE_VIEW = "azure_pyannote_forced_ar"
AR_RE = re.compile(r"[\u0600-\u06ff]")
TOKEN_RE = re.compile(r"[\u0600-\u06ff]+|[A-Za-z0-9']+|[^\w\s]", re.UNICODE)
TASHKEEL = {chr(cp) for cp in range(0x0610, 0x061B)} | {chr(cp) for cp in range(0x064B, 0x0660)} | {chr(0x0670)}
FOLD = str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ٱ":"ا","ؤ":"و","ئ":"ي","ى":"ي","ة":"ه"})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo_root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=True).stdout.strip()


def strip_tashkeel(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFC", text) if ch not in TASHKEEL and unicodedata.category(ch) != "Mn")


def skeleton(text: str) -> str:
    return strip_tashkeel(text).translate(FOLD).casefold()


def is_arabic(text: str) -> bool:
    return bool(AR_RE.search(text))


def token_text(token: dict[str, Any]) -> str:
    return str(token.get("text") or token.get("surface") or token.get("normalized") or "").strip()


def token_center(token: dict[str, Any]) -> float | None:
    for name in ("center", "global_center"):
        value = token.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    start = token.get("global_start", token.get("start"))
    end = token.get("global_end", token.get("end"))
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return (float(start) + float(end)) / 2
    return None


def segment_span(row: dict[str, Any]) -> tuple[float, float]:
    start = float(row.get("segment_start", row.get("global_start", row.get("start", 0.0))))
    end = float(row.get("segment_end", row.get("global_end", row.get("end", start + float(row.get("duration", 20.0))))))
    return start, end


def azure_observations(parent: dict[str, Any], seg_start: float, seg_end: float) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    children = parent.get("children", []) if isinstance(parent.get("children"), list) else []
    if not children:
        text = str(parent.get("azure_forced_ar_text") or parent.get("text") or "").strip()
        children = [{"transcript": text, "global_start": seg_start, "global_end": seg_end}] if text else []
    for child_index, child in enumerate(children):
        text = str(child.get("transcript") or child.get("text") or "").strip()
        pieces = TOKEN_RE.findall(text)
        if not pieces:
            continue
        start = child.get("global_start")
        end = child.get("global_end")
        start = float(start) if isinstance(start, (int, float)) else seg_start
        end = float(end) if isinstance(end, (int, float)) else seg_end
        if end < start:
            start, end = end, start
        width = max(0.001, end - start)
        for position, surface in enumerate(pieces):
            local_start = start + width * position / len(pieces)
            local_end = start + width * (position + 1) / len(pieces)
            observations.append({
                "text": surface,
                "normalized": skeleton(surface),
                "global_start": round(local_start, 6),
                "global_end": round(local_end, 6),
                "center": round((local_start + local_end) / 2, 6),
                "view": AZURE_VIEW,
                "child_index": child_index,
                "child_token_position": position,
            })
    return observations


def choose_match(tokens: list[dict[str, Any]], obs: dict[str, Any], max_seconds: float) -> int | None:
    obs_key = skeleton(obs["text"])
    obs_center = float(obs["center"])
    candidates: list[tuple[int, float, int]] = []
    for index, token in enumerate(tokens):
        center = token_center(token)
        if center is None or abs(center - obs_center) > max_seconds:
            continue
        surface = token_text(token)
        same = skeleton(surface) == obs_key and bool(obs_key)
        script_upgrade = is_arabic(obs["text"]) and not is_arabic(surface)
        if same or script_upgrade:
            candidates.append((0 if same else 1, abs(center - obs_center), index))
    return min(candidates)[2] if candidates else None


def corroborate(token: dict[str, Any], obs: dict[str, Any]) -> None:
    views = list(dict.fromkeys([*token.get("views", []), AZURE_VIEW]))
    token["views"] = views
    observations = list(token.get("observations", []))
    observations.append(obs)
    token["observations"] = observations
    token["support_count"] = len(views)
    flags = list(dict.fromkeys(token.get("flags", [])))
    surface = token_text(token)
    if is_arabic(obs["text"]) and not is_arabic(surface):
        flags.append("script_disagreement")
        token["text"] = obs["text"]
        token["surface"] = obs["text"]
    token["flags"] = list(dict.fromkeys(flags))
    if token.get("acceptance_tier") == "C_single_witness" and len(views) >= 2:
        token["acceptance_tier"] = "A_corroborated"


def azure_only_token(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": obs["text"],
        "surface": obs["text"],
        "normalized": skeleton(obs["text"]),
        "global_start": obs["global_start"],
        "global_end": obs["global_end"],
        "center": obs["center"],
        "views": [AZURE_VIEW],
        "support_count": 1,
        "acceptance_tier": "C_single_witness",
        "alternates": [],
        "flags": ["azure_only", "needs_vocalization"],
        "observations": [obs],
    }


def ar_letters(text: str) -> int:
    return sum(ch.isalpha() and is_arabic(ch) for ch in strip_tashkeel(text))


def tashkeel_density(text: str) -> float:
    letters = sum(ch.isalpha() and is_arabic(ch) for ch in strip_tashkeel(text))
    marks = sum(ch in TASHKEEL or unicodedata.category(ch) == "Mn" for ch in text)
    return marks / letters if letters else 0.0


def text_from_tokens(tokens: list[dict[str, Any]]) -> str:
    return " ".join(token_text(token) for token in tokens if token_text(token)).strip()


def seg_id(position: int) -> str:
    return f"seg_{position:06d}"


def immediate_duplicate_count(rows: list[dict[str, Any]], n: int = 6) -> int:
    count = 0
    for row in rows:
        values = [skeleton(token_text(token)) for token in row.get("tokens", []) if token_text(token)]
        for i in range(max(0, len(values) - 2*n + 1)):
            if values[i:i+n] == values[i+n:i+2*n]:
                count += 1
    return count


def recall(hyp: str, ref: str, arabic: bool) -> float:
    def selected(text: str) -> set[str]:
        values = {skeleton(piece) for piece in TOKEN_RE.findall(text)}
        return {value for value in values if value and (is_arabic(value) if arabic else not is_arabic(value))}
    gold = selected(ref)
    return len(selected(hyp) & gold) / len(gold) if gold else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Integrate Azure/Pyannote as the lowest-priority fifth view in repaired Silver v3.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--silver-v3", type=Path, required=True)
    parser.add_argument("--azure-parent", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-align-seconds", type=float, default=0.85)
    parser.add_argument("--pilot-segments", type=int, default=30)
    parser.add_argument("--min-pilot-ar-recall", type=float, default=0.33)
    parser.add_argument("--min-pilot-en-recall", type=float, default=0.80)
    parser.add_argument("--min-tashkeel-density", type=float, default=0.20)
    parser.add_argument("--max-immediate-duplicates", type=int, default=1)
    args = parser.parse_args()

    config = {
        "max_align_seconds": args.max_align_seconds,
        "pilot_segments": args.pilot_segments,
        "min_pilot_ar_recall": args.min_pilot_ar_recall,
        "min_pilot_en_recall": args.min_pilot_en_recall,
        "min_tashkeel_density": args.min_tashkeel_density,
        "max_immediate_duplicates": args.max_immediate_duplicates,
        "view_priority": ["canonical_20s","whole_slice","context_10s_stride_5s","local_2p5s_contiguous",AZURE_VIEW],
        "azure_vote_key": "tashkeel_stripped_light_orthographic_fold",
    }
    config_hash = sha256_bytes(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
    commit = git_head(args.repo_root)
    silver_rows = read_jsonl(args.silver_v3)
    azure_rows = read_jsonl(args.azure_parent)
    gold_rows = read_jsonl(args.gold)
    if not (len(silver_rows) == len(azure_rows) == len(gold_rows)):
        raise ValueError(f"Row count mismatch silver={len(silver_rows)} azure={len(azure_rows)} gold={len(gold_rows)}")

    output_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    unmatched = 0
    corroborated = 0
    script_upgrades = 0
    before_counts: list[int] = []
    after_counts: list[int] = []

    for position, (silver, azure, gold) in enumerate(zip(silver_rows, azure_rows, gold_rows)):
        row = copy.deepcopy(silver)
        tokens = list(row.get("tokens", []))
        seg_start, seg_end = segment_span(row)
        parakeet_text = text_from_tokens(tokens) or str(row.get("text") or "")
        for obs in azure_observations(azure, seg_start, seg_end):
            match = choose_match(tokens, obs, args.max_align_seconds)
            if match is None:
                tokens.append(azure_only_token(obs))
                unmatched += 1
            else:
                was_latin = not is_arabic(token_text(tokens[match]))
                corroborate(tokens[match], obs)
                corroborated += 1
                if was_latin and is_arabic(token_text(tokens[match])):
                    script_upgrades += 1
        tokens.sort(key=lambda token: (float(token_center(token) or seg_start), str(token.get("word_position", ""))))
        integrated_text = text_from_tokens(tokens)
        before_counts.append(ar_letters(parakeet_text))
        after_counts.append(ar_letters(integrated_text))
        row.update({
            "schema_version": "silver_plus_v3_fifth_view_segment_v1",
            "segment_position": position,
            "segment_index": position,
            "segment_id": str(row.get("segment_id")),
            "embedded_seg_id": seg_id(position),
            "silver_v3_text": parakeet_text,
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
                "segment_id": row["segment_id"],
                "embedded_seg_id": seg_id(position),
                "token_position": token_position,
                "text": token_text(token),
                "views": token.get("views", []),
                "support_count": token.get("support_count", len(token.get("views", []))),
                "tier": token.get("acceptance_tier"),
                "alternates": token.get("alternates", []),
                "flags": token.get("flags", []),
                "observations": token.get("observations", []),
            })

    no_regression_segments = [i for i, (before, after) in enumerate(zip(before_counts, after_counts)) if after < before]
    full_text = " ".join(row["silver_plus_v3_text"] for row in output_rows)
    pilot_hyp = " ".join(row["silver_plus_v3_text"] for row in output_rows[:args.pilot_segments])
    pilot_gold = " ".join(str(row.get("gold_text") or row.get("text") or row.get("transcript") or "") for row in gold_rows[:args.pilot_segments])
    tier_counts = Counter(token.get("acceptance_tier", "unknown") for row in output_rows for token in row.get("tokens", []))
    total_tokens = sum(tier_counts.values())
    a_ratio = tier_counts.get("A_corroborated", 0) / total_tokens if total_tokens else 0.0
    duplicates = immediate_duplicate_count(output_rows)
    density = tashkeel_density(full_text)
    pilot_ar = recall(pilot_hyp, pilot_gold, True)
    pilot_en = recall(pilot_hyp, pilot_gold, False)
    seg_mapping_ok = all(row["embedded_seg_id"] == seg_id(i) for i, row in enumerate(output_rows))
    all_views = {view for row in output_rows for token in row.get("tokens", []) for view in token.get("views", [])}
    required_views = {"canonical_20s","whole_slice","context_10s_stride_5s","local_2p5s_contiguous",AZURE_VIEW}
    gates = {
        "azure_used_inside_silver_v3_lattice": True,
        "no_ar_letter_regression": not no_regression_segments,
        "tashkeel_density_pass": density >= args.min_tashkeel_density,
        "pilot_ar_recall_pass": pilot_ar >= args.min_pilot_ar_recall,
        "pilot_en_recall_pass": pilot_en >= args.min_pilot_en_recall,
        "duplicate_budget_pass": duplicates <= args.max_immediate_duplicates,
        "all_views_present": required_views <= all_views,
        "seg_id_mapping_pass": seg_mapping_ok,
        "repository_commit_stamped": bool(commit),
        "config_hash_stamped": bool(config_hash),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_path = args.output_dir / f"{args.output_prefix}_segment_level.jsonl"
    provenance_path = args.output_dir / f"{args.output_prefix}_token_provenance.jsonl"
    report_path = args.output_dir / f"{args.output_prefix}_quality_report.json"
    json_path = args.output_dir / f"{args.output_prefix}.json"
    package_path = args.output_dir / f"{args.output_prefix}.zip"
    write_jsonl(segment_path, output_rows)
    write_jsonl(provenance_path, provenance_rows)
    json_path.write_text(json.dumps({"schema_version":"silver_plus_v3_fifth_view_export_v1","repository_commit":commit,"config_hash":config_hash,"segments":output_rows}, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    report = {
        "schema_version": "silver_plus_v3_fifth_view_quality_v1",
        "repository_commit": commit,
        "config": config,
        "config_hash": config_hash,
        "segment_count": len(output_rows),
        "tier_distribution": dict(tier_counts),
        "a_tier_ratio": round(a_ratio, 6),
        "azure": {"corroborated_tokens": corroborated, "azure_only_tokens": unmatched, "script_upgrades": script_upgrades},
        "before_after": {
            "parakeet_ar_letters": sum(before_counts),
            "integrated_ar_letters": sum(after_counts),
            "segments_losing_ar_letters": no_regression_segments,
            "tashkeel_density": round(density, 6),
        },
        "pilot_window": {"segment_count": min(args.pilot_segments, len(output_rows)), "ar_recall": round(pilot_ar, 6), "en_recall": round(pilot_en, 6)},
        "immediate_duplicate_6gram_count": duplicates,
        "views_present": sorted(all_views),
        "gates": gates,
        "passed": all(gates.values()),
        "outputs": {"segment_jsonl":str(segment_path),"token_provenance_jsonl":str(provenance_path),"json":str(json_path),"zip":str(package_path)},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in (segment_path, provenance_path, json_path, report_path):
            archive.write(path, arcname=path.name)
        manifest = {
            "schema_version":"silver_plus_v3_package_manifest_v1",
            "repository_commit":commit,
            "config_hash":config_hash,
            "files": [{"filename":p.name,"sha256":sha256_file(p),"size_bytes":p.stat().st_size} for p in (segment_path, provenance_path, json_path, report_path)],
        }
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
