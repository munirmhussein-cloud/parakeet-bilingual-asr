from __future__ import annotations

import unicodedata
from copy import deepcopy
from typing import Any, Iterable

VALID_LANGUAGES = {"ar-AR", "en-US"}


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_arabic_letter(character: str) -> bool:
    if not _is_letter(character):
        return False

    codepoint = ord(character)

    return (
        0x0600 <= codepoint <= 0x06FF
        or 0x0750 <= codepoint <= 0x077F
        or 0x08A0 <= codepoint <= 0x08FF
        or 0xFB50 <= codepoint <= 0xFDFF
        or 0xFE70 <= codepoint <= 0xFEFF
    )


def _is_latin_letter(character: str) -> bool:
    if not _is_letter(character):
        return False

    try:
        return "LATIN" in unicodedata.name(character)
    except ValueError:
        return False


def detect_lexical_language(text: str) -> str | None:
    """Detect language from lexical letters only.

    Punctuation, symbols, whitespace, and digit-only tokens return None so
    their language can be inherited from surrounding lexical tokens.
    """
    text = str(text or "")

    arabic_letters = sum(
        1 for character in text
        if _is_arabic_letter(character)
    )
    latin_letters = sum(
        1 for character in text
        if _is_latin_letter(character)
    )

    if arabic_letters > latin_letters:
        return "ar-AR"

    if latin_letters > arabic_letters:
        return "en-US"

    if arabic_letters:
        return "ar-AR"

    if latin_letters:
        return "en-US"

    return None


def assign_languages_with_inheritance(
    items: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign lexical languages and inherit punctuation language.

    Non-lexical tokens inherit from the closest previous lexical token.
    If none exists, they inherit from the closest following lexical token.
    If a sequence contains no lexical tokens, a valid original language is
    preserved as a documented fallback.
    """
    tagged = [deepcopy(item) for item in items]

    lexical_languages = [
        detect_lexical_language(item.get("corrected_text", ""))
        for item in tagged
    ]

    previous_languages: list[str | None] = []
    previous_language: str | None = None

    for language in lexical_languages:
        if language:
            previous_language = language
        previous_languages.append(previous_language)

    following_languages: list[str | None] = [None] * len(tagged)
    following_language: str | None = None

    for index in range(len(tagged) - 1, -1, -1):
        if lexical_languages[index]:
            following_language = lexical_languages[index]
        following_languages[index] = following_language

    for index, item in enumerate(tagged):
        detected = lexical_languages[index]

        if detected:
            language = detected
            method = "script_detected"
        elif previous_languages[index]:
            language = previous_languages[index]
            method = "inherited_previous"
        elif following_languages[index]:
            language = following_languages[index]
            method = "inherited_next"
        else:
            original = item.get("selected_language")
            language = (
                original
                if original in VALID_LANGUAGES
                else "ar-AR"
            )
            method = "original_fallback"

        item["selected_language"] = language
        item["language_assignment_method"] = method
        item["language_assignment_source"] = "automatic_script_v1"

        training_span = item.get("training_span")
        if isinstance(training_span, dict):
            training_span = deepcopy(training_span)
            training_span["language"] = language
            item["training_span"] = training_span

    return tagged
