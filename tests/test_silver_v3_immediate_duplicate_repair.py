from __future__ import annotations

from copy import deepcopy

from pipeline.silver_v3.repair_immediate_duplicates import (
    collect_evidence_ids,
    immediate_duplicate_cases,
    refresh_row,
    repair_rows,
)


PUNCTUATION = {".", ",", "!", "?", ":", ";"}
VIEW = "context_10s_stride_5s"


def token(text: str, center: float, index: int) -> dict:
    normalized = "" if text in PUNCTUATION else text.casefold()
    observation = {
        "observation_id": f"{VIEW}|source_{index}|{index}",
        "view": VIEW,
        "source_id": f"source_{index}",
        "source_token_index": index,
        "surface": text,
        "normalized": normalized,
        "center": center,
        "start": center,
        "end": center,
        "timing_method": "raw_global_timestamp",
    }
    return {
        "slot_id": index,
        "center": center,
        "text": text,
        "normalized": normalized,
        "acceptance_tier": "C_single_witness",
        "single_witness": True,
        "witness_view_count": 1,
        "views": [VIEW],
        "observation_count": 1,
        "observation_ids": [observation["observation_id"]],
        "observations": [observation],
        "alternates": [],
    }


def row(
    segment_position: int,
    phrase: list[str],
    *,
    overlaid: bool,
    observation_base: int,
) -> dict:
    first = []
    second = []
    for offset, surface in enumerate(phrase):
        center = 10.0 + offset * 0.4
        first.append(
            token(surface, center, observation_base + offset)
        )
        second.append(
            token(
                surface,
                center if overlaid else center + 4.0,
                observation_base + 100 + offset,
            )
        )
    output = {
        "schema_version": "silver_v3_segment_level_v4",
        "lecture_id": "lecture_002",
        "segment_position": segment_position,
        "segment_index": segment_position,
        "segment_id": (
            "lecture_002__canonical_20s__"
            f"{segment_position:05d}"
        ),
        "segment_start": segment_position * 20.0,
        "segment_end": segment_position * 20.0 + 20.0,
        "duration": 20.0,
        "silver_text": "",
        "has_silver_text": True,
        "token_count": 12,
        "tier_counts": {},
        "view_contribution_counts": {},
        "immediate_duplicate_6gram_count": 1,
        "duplicate_overlap_collapses": [],
        "duplicate_overlap_collapse_count": 0,
        "tokens": first + second,
    }
    refresh_row(output)
    return output


def test_validator_semantics_include_punctuation() -> None:
    tokens = row(
        46,
        ["the", "chest", "to", "the", "navel", "."],
        overlaid=True,
        observation_base=0,
    )["tokens"]

    cases = immediate_duplicate_cases(tokens)

    assert len(cases) == 1
    assert cases[0]["ngram"] == "the chest to the navel ."
    assert cases[0]["first_positions"] == [0, 1, 2, 3, 4, 5]
    assert cases[0]["second_positions"] == [6, 7, 8, 9, 10, 11]


def test_single_allowed_duplicate_is_not_rewritten() -> None:
    rows = [
        row(
            62,
            ["they", "were", "people", "of", "quality", "."],
            overlaid=True,
            observation_base=0,
        )
    ]
    original = deepcopy(rows)

    repair = repair_rows(rows, maximum_remaining=1)

    assert repair["before_count"] == 1
    assert repair["after_count"] == 1
    assert repair["collapse_count"] == 0
    assert rows == original


def test_repairs_two_overlaid_cases_and_retains_one_allowed_case() -> None:
    rows = [
        row(
            46,
            ["the", "chest", "to", "the", "navel", "."],
            overlaid=True,
            observation_base=0,
        ),
        row(
            154,
            ["by", "the", "name", "of", "zahir", ","],
            overlaid=True,
            observation_base=1000,
        ),
        row(
            183,
            [".", "it", "was", "too", "late", "."],
            overlaid=False,
            observation_base=2000,
        ),
    ]
    evidence_before = collect_evidence_ids(rows)

    repair = repair_rows(rows, maximum_remaining=1)

    assert repair["before_count"] == 3
    assert repair["collapse_count"] == 2
    assert repair["after_count"] == 1
    assert repair["within_quality_limit"] is True
    assert repair["evidence_accounting_preserved"] is True
    assert collect_evidence_ids(rows) == evidence_before
    assert [item["token_count"] for item in rows] == [6, 6, 12]
    assert [item["immediate_duplicate_6gram_count"] for item in rows] == [0, 0, 1]
    assert repair["remaining_cases"][0]["segment_position"] == 183
    assert repair["remaining_cases"][0]["eligible"] is False


def test_pair_view_mismatch_is_not_collapsed() -> None:
    rows = [
        row(
            46,
            ["the", "chest", "to", "the", "navel", "."],
            overlaid=True,
            observation_base=0,
        ),
        row(
            154,
            ["by", "the", "name", "of", "zahir", ","],
            overlaid=False,
            observation_base=1000,
        ),
    ]
    rows[0]["tokens"][6]["views"] = ["canonical_20s"]

    repair = repair_rows(rows, maximum_remaining=1)

    assert repair["before_count"] == 2
    assert repair["collapse_count"] == 0
    assert repair["after_count"] == 2
    assert repair["within_quality_limit"] is False
