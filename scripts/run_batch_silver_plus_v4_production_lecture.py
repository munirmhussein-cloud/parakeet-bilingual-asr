from __future__ import annotations

"""Run the accepted Silver+ v4 producer with safe vocalization upgrades.

The upgrade is deliberately narrow: an Azure observation may replace a token
surface only when removing combining marks leaves the exact same Unicode base
letters and the observation contains strictly more vocalization marks. No
letter folding, fuzzy matching, or threshold changes are used.
"""

import unicodedata
from typing import Any

import integrate_silver_plus_v3_fifth_view as integration
import run_silver_plus_v3_fifth_view_export_v2 as integration_v3
import run_silver_plus_v4_production_lecture as producer


_ORIGINAL_CORROBORATE = integration_v3.corroborate


def exact_base_surface(text: str) -> str:
    """Remove combining marks while preserving every base character exactly."""

    decomposed = unicodedata.normalize("NFD", str(text or ""))
    bare = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
        and character not in integration.TASHKEEL
    )
    return unicodedata.normalize("NFC", bare)


def vocalization_mark_count(text: str) -> int:
    return sum(
        unicodedata.category(character) == "Mn"
        or character in integration.TASHKEEL
        for character in unicodedata.normalize("NFD", str(text or ""))
    )


def corroborate_with_exact_base_vocalization(
    token: dict[str, Any],
    observation: dict[str, Any],
) -> None:
    """Corroborate normally, then prefer a safer, more vocalized surface."""

    _ORIGINAL_CORROBORATE(token, observation)

    current = integration.token_text(token)
    candidate = str(
        observation.get("text")
        or observation.get("surface")
        or ""
    ).strip()

    if not current or not candidate:
        return
    if not integration.is_arabic(current) or not integration.is_arabic(candidate):
        return
    if exact_base_surface(current) != exact_base_surface(candidate):
        return
    if vocalization_mark_count(candidate) <= vocalization_mark_count(current):
        return

    token["text"] = candidate
    token["surface"] = candidate
    token["normalized"] = integration.skeleton(candidate)
    token["flags"] = list(
        dict.fromkeys(
            [
                *token.get("flags", []),
                "exact_base_vocalization_upgrade",
            ]
        )
    )


def main() -> int:
    integration_v3.corroborate = corroborate_with_exact_base_vocalization
    return producer.main()


if __name__ == "__main__":
    raise SystemExit(main())
