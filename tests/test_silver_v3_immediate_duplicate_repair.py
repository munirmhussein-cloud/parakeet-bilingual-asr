from __future__ import annotations

from copy import deepcopy

from pipeline.silver_v3.repair_immediate_duplicates import (
    collect_evidence_ids,
    immediate_duplicate_cases,
    refresh_row,
    repair_rows,
    scan_rows,
)


PUNCTUATION = {".", ",", "!", "?", ":", ";"}


def token(
    text: str,
    center: float,
    views: list[str],
    index: int,
) -> dict:
    normalized = "" if text in PUNCTUATION else text.casefold()
    observations = [
        {
            "observation_id": f"{view}|source_{index}_{offset}|{index}",
            "view": view,
            "source_id": f"source_{index}_{offset}",
            "source_token_index": index,
            "surface": text,
            "normalized": normalized,
            "center": center,
            "start": center,
            "end": center,
            "timing_method": "raw_global_timestamp",
        }
        for offset, view in enumerate(views)
    ]
    return {
        "slot_id": index,
        "center": center,
        "text": text,
        "normalized": normalized,
        "acceptance_tier": (
            "A_corroborated"
            if len(views) >= 2
            else "C_single_witness"
        ),
        "single_witness": len(views) < 2,
        "witness_view_count": len(views),
        "views": views,
        "observation_count": len(observations),
        "observation_ids": [
            item["observation_id"]
            for item in observations
        ],
        "observations": observations,
        "alternates": [],
    }


def row(
    segment_position: int,
    phrase: list[str],
    first_centers: list[float],
    second_centers: list[float],
    first_views: list[list[str]],
    second_views: list[list[str]],
    observation_base: int,
) -> dict:
    first = [
        token(
            surface,
            center,
            views,
            observation_base + offset,
        )
        for offset, (surface, center, views) in enumerate(
            zip(phrase, first_centers, first_views)
        )
    ]
    second = [
        token(
            surface,
            center,
            views,
            observation_base + 100 + offset,
        )
        for offset, (surface, center, views) in enumerate(
            zip(phrase, second_centers, second_views)
        )
    ]
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


def lecture_002_rows() -> list[dict]:
    canonical = "canonical_20s"
    context = "context_10s_stride_5s"
    local = "local_2p5s_contiguous"
    whole = "whole_slice"

    return [
        row(
            46,
            ["the", "chest", "to", "the", "navel", "."],
            [936.06, 936.26, 936.50, 936.62, 936.90, 937.04],
            [937.53, 937.75, 938.00, 938.04, 938.27, 938.41],
            [
                [context, local, whole],
                [context, local, whole],
                [context, local, whole],
                [context, local, whole],
                [context, local, whole],
                [context, local],
            ],
            [
                [canonical, context, local, whole],
                [canonical, context, local, whole],
                [canonical, context, local, whole],
                [canonical, context, local, whole],
                [canonical, context, local, whole],
                [canonical, context, local, whole],
            ],
            0,
        ),
        row(
            154,
            ["by", "the", "name", "of", "zahir", ","],
            [3080.72, 3080.96, 3081.04, 3081.16, 3082.00, 3082.245],
            [3082.72, 3082.80, 3082.96, 3083.09, 3083.52, 3083.76],
            [[canonical, context, whole]] * 6,
            [
                [canonical, context, whole],
                [canonical, context, local, whole],
                [canonical, context, whole],
                [canonical, context, local, whole],
                [canonical, context, local, whole],
                [canonical, context, local, whole],
            ],
            1000,
        ),
        row(
            183,
            [".", "it", "was", "too", "late", "."],
            [3672.72, 3673.04, 3673.265, 3673.42, 3673.64, 3673.68],
            [3673.76, 3674.14, 3674.30, 3674.48, 3674.70, 3674.75],
            [
                [canonical, context],
                [canonical, context],
                [canonical, context, local],
                [canonical, context],
                [canonical, context, local, whole],
                [canonical, context],
            ],
            [
                [whole],
                [canonical, context, whole],
                [canonical, context, whole],
                [canonical, context, whole],
                [canonical, context, whole],
                [canonical, context, local, whole],
            ],
            2000,
        ),
    ]


def test_validator_semantics_include_punctuation() -> None:
    cases = immediate_duplicate_cases(
        lecture_002_rows()[0]["tokens"]
    )
    assert len(cases) == 1
    assert cases[0]["ngram"] == "the chest to the navel ."


def test_single_allowed_duplicate_is_not_rewritten() -> None:
    rows = lecture_002_rows()[:1]
    original = deepcopy(rows)
    result = repair_rows(rows, maximum_remaining=1)
    assert result["before_count"] == 1
    assert result["after_count"] == 1
    assert result["collapse_count"] == 0
    assert rows == original


def test_real_lecture_002_cases_are_rigid_translation_candidates() -> None:
    cases = scan_rows(lecture_002_rows())
    assert len(cases) == 3
    assert all(case["eligible"] is True for case in cases)

    by_position = {
        case["segment_position"]: case
        for case in cases
    }
    assert by_position[46]["pair_center_delta_spread"] < 0.14
    assert by_position[154]["pair_center_delta_spread"] < 0.49
    assert by_position[183]["pair_center_delta_spread"] < 0.07
    assert by_position[183]["minimum_pair_view_jaccard"] == 0.5


def test_real_lecture_002_reduces_three_to_one() -> None:
    rows = lecture_002_rows()
    evidence_before = collect_evidence_ids(rows)

    result = repair_rows(rows, maximum_remaining=1)

    assert result["before_count"] == 3
    assert result["collapse_count"] == 2
    assert result["after_count"] == 1
    assert result["within_quality_limit"] is True
    assert result["evidence_accounting_preserved"] is True
    assert collect_evidence_ids(rows) == evidence_before

    # The two most rigid phrase loops are repaired; the least rigid
    # remaining case is permitted by the unchanged maximum of one.
    assert [
        item["segment_position"]
        for item in result["collapses"]
    ] == [183, 46]
    assert result["remaining_cases"][0]["segment_position"] == 154
    assert [item["token_count"] for item in rows] == [6, 12, 6]


def test_nonrigid_or_disjoint_evidence_is_not_repaired() -> None:
    rows = lecture_002_rows()[:1]

    for offset, position in enumerate(range(6, 12)):
        rows[0]["tokens"][position]["center"] += offset * 0.30

    case = scan_rows(rows)[0]
    assert case["eligible"] is False
    assert any(
        "delta_spread_exceeds_limit" in reason
        for reason in case["ineligible_reasons"]
    )

    rows = lecture_002_rows()[:1]
    for position in range(6, 11):
        rows[0]["tokens"][position]["views"] = [
            "azure_pyannote_forced_ar"
        ]

    case = scan_rows(rows)[0]
    assert case["eligible"] is False
    assert any(
        "view_overlap_below_limit" in reason
        for reason in case["ineligible_reasons"]
    )
