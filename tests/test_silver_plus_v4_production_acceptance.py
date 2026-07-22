from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_silver_plus_v4_production_lecture as producer


def token(text: str) -> dict:
    views = list(producer.REQUIRED_VIEWS)
    return {
        "text": text,
        "normalized": producer.skeleton(text),
        "views": views,
        "acceptance_tier": "A_corroborated",
        "observations": [
            {
                "view": producer.AZURE_VIEW,
                "text": text,
            }
        ],
    }


def test_finalize_segment_row_flags_empty_text() -> None:
    row = {
        "tokens": [],
        "silver_v3_text": "",
        "embedded_seg_id": "seg_000001",
    }
    fixed = producer.finalize_segment_row(row)
    assert fixed["silver_plus_v4_text"] == ""
    assert fixed["has_silver_plus_v4_text"] is False
    assert fixed["immediate_duplicate_6gram_count"] == 0


def test_production_validation_emits_real_seven_gate_stamp() -> None:
    tokens = [token("بِسْمِ"), token("اللهِ"), token("hello")]
    row = {
        "segment_id": "lecture_001__canonical_20s__00000",
        "embedded_seg_id": "seg_000000",
        "silver_v3_text": "بِسْمِ اللهِ hello",
        "silver_plus_v4_text": "بِسْمِ اللهِ hello",
        "tokens": tokens,
        "immediate_duplicate_6gram_count": 0,
    }
    provenance = [
        {
            "tier": "A_corroborated",
            "text": item["text"],
        }
        for item in tokens
    ]
    azure_rows = [
        {
            "children": [
                {
                    "transcript": "بسم الله hello",
                }
            ]
        }
    ]

    validation = producer.run_production_gates(
        [row],
        provenance,
        azure_rows,
        True,
    )

    assert validation["validation_performed"] is True
    assert validation["all_gates_pass"] is True
    assert set(validation["gates"]) == {
        "all_views_present",
        "corroboration_floor",
        "determinism",
        "duplicate_budget",
        "no_regression_vs_silver_v3",
        "pilot_window_union_reproduction",
        "vocalization_floor",
    }
    assert all(
        gate["pass"] is True
        for gate in validation["gates"].values()
    )


def test_all_gates_pass_is_the_boolean_conjunction() -> None:
    tokens = [token("بِسْمِ"), token("اللهِ"), token("hello")]
    row = {
        "segment_id": "lecture_001__canonical_20s__00000",
        "embedded_seg_id": "seg_000000",
        "silver_v3_text": "بِسْمِ اللهِ hello",
        "silver_plus_v4_text": "بِسْمِ اللهِ hello",
        "tokens": tokens,
        "immediate_duplicate_6gram_count": 0,
    }
    provenance = [
        {
            "tier": "A_corroborated",
            "text": item["text"],
        }
        for item in tokens
    ]
    azure_rows = [
        {
            "children": [
                {
                    "transcript": "بسم الله hello",
                }
            ]
        }
    ]

    validation = producer.run_production_gates(
        [row],
        provenance,
        azure_rows,
        False,
    )

    assert validation["gates"]["determinism"]["pass"] is False
    assert validation["all_gates_pass"] is False
