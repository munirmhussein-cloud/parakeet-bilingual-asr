from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_batch_silver_plus_v4_production_lecture as cleanroom


def observation(text: str) -> dict:
    return {
        "text": text,
        "normalized": cleanroom.integration.skeleton(text),
        "global_start": 0.0,
        "global_end": 0.1,
        "center": 0.05,
        "view": cleanroom.integration.AZURE_VIEW,
    }


def token(text: str) -> dict:
    return {
        "text": text,
        "surface": text,
        "normalized": cleanroom.integration.skeleton(text),
        "views": ["canonical_20s"],
        "support_count": 1,
        "acceptance_tier": "C_single_witness",
        "flags": [],
        "observations": [],
    }


def test_exact_base_vocalization_upgrade() -> None:
    value = token("الله")
    cleanroom.corroborate_with_exact_base_vocalization(
        value,
        observation("اللّٰهِ"),
    )
    assert value["text"] == "اللّٰهِ"
    assert value["surface"] == "اللّٰهِ"
    assert "exact_base_vocalization_upgrade" in value["flags"]
    assert cleanroom.integration.AZURE_VIEW in value["views"]


def test_exact_base_rule_does_not_fold_letters() -> None:
    value = token("رحمة")
    cleanroom.corroborate_with_exact_base_vocalization(
        value,
        observation("رَحْمَه"),
    )
    assert value["text"] == "رحمة"
    assert "exact_base_vocalization_upgrade" not in value["flags"]


def test_exact_base_rule_never_downgrades_vocalization() -> None:
    value = token("بِسْمِ")
    cleanroom.corroborate_with_exact_base_vocalization(
        value,
        observation("بسم"),
    )
    assert value["text"] == "بِسْمِ"
    assert "exact_base_vocalization_upgrade" not in value["flags"]
