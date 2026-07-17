from scripts.integrate_silver_plus_v3_fifth_view import (
    AZURE_VIEW,
    azure_only_token,
    corroborate,
    seg_id,
    skeleton,
)


def test_skeleton_folds_tashkeel_and_arabic_variants() -> None:
    assert skeleton("إِلَى") == skeleton("الى")
    assert skeleton("رَحْمَة") == skeleton("رحمه")


def test_azure_corroboration_preserves_vocalized_parakeet_surface() -> None:
    token = {
        "text": "نَعُوذُ",
        "surface": "نَعُوذُ",
        "views": ["canonical_20s"],
        "acceptance_tier": "C_single_witness",
        "observations": [],
        "flags": [],
    }
    observation = {
        "text": "نعوذ",
        "view": AZURE_VIEW,
        "center": 1.0,
        "global_start": 0.9,
        "global_end": 1.1,
    }

    corroborate(token, observation)

    assert token["text"] == "نَعُوذُ"
    assert token["surface"] == "نَعُوذُ"
    assert AZURE_VIEW in token["views"]
    assert token["acceptance_tier"] == "A_corroborated"


def test_azure_only_token_is_flagged_for_vocalization() -> None:
    token = azure_only_token(
        {
            "text": "نعوذ",
            "normalized": "نعوذ",
            "center": 1.0,
            "global_start": 0.9,
            "global_end": 1.1,
            "view": AZURE_VIEW,
        }
    )

    assert token["acceptance_tier"] == "C_single_witness"
    assert token["views"] == [AZURE_VIEW]
    assert "azure_only" in token["flags"]
    assert "needs_vocalization" in token["flags"]


def test_embedded_segment_mapping_is_deterministic() -> None:
    assert seg_id(0) == "seg_000000"
    assert seg_id(62) == "seg_000062"
    assert seg_id(137) == "seg_000137"
