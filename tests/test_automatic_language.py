from src.annotation.automatic_language import (
    assign_languages_with_inheritance,
    detect_lexical_language,
)


def make_item(text, language="ar-AR"):
    return {
        "corrected_text": text,
        "selected_language": language,
        "training_span": {
            "language": language,
        },
    }


def test_detect_lexical_language():
    assert detect_lexical_language("Hello") == "en-US"
    assert detect_lexical_language("مرحبا") == "ar-AR"
    assert detect_lexical_language(",") is None
    assert detect_lexical_language("؟") is None
    assert detect_lexical_language("123") is None


def test_punctuation_inherits_previous_language():
    items = [
        make_item("Hello"),
        make_item(","),
        make_item("world"),
    ]

    tagged = assign_languages_with_inheritance(items)

    assert tagged[0]["selected_language"] == "en-US"
    assert tagged[1]["selected_language"] == "en-US"
    assert (
        tagged[1]["language_assignment_method"]
        == "inherited_previous"
    )
    assert tagged[2]["selected_language"] == "en-US"


def test_leading_punctuation_inherits_next_language():
    items = [
        make_item("("),
        make_item("مرحبا"),
    ]

    tagged = assign_languages_with_inheritance(items)

    assert tagged[0]["selected_language"] == "ar-AR"
    assert (
        tagged[0]["language_assignment_method"]
        == "inherited_next"
    )


def test_code_switch_punctuation_uses_previous_token():
    items = [
        make_item("Hello"),
        make_item(","),
        make_item("كيف"),
        make_item("؟"),
    ]

    tagged = assign_languages_with_inheritance(items)

    assert [
        item["selected_language"]
        for item in tagged
    ] == [
        "en-US",
        "en-US",
        "ar-AR",
        "ar-AR",
    ]
