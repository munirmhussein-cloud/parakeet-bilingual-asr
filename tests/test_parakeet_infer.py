from types import SimpleNamespace

from src.inference.parakeet_infer import _extract_words_from_riva_response, _word_seconds


def test_word_seconds_from_duration_like_object():
    value = SimpleNamespace(seconds=1, nanos=250_000_000)
    assert _word_seconds(value) == 1.25


def test_extract_words_from_riva_response():
    response = SimpleNamespace(
        results=[
            SimpleNamespace(
                alternatives=[
                    SimpleNamespace(
                        transcript="hello world",
                        words=[
                            SimpleNamespace(
                                word="hello",
                                start_time=SimpleNamespace(seconds=0, nanos=100_000_000),
                                end_time=SimpleNamespace(seconds=0, nanos=400_000_000),
                                confidence=0.9,
                            )
                        ],
                    )
                ]
            )
        ]
    )

    words = _extract_words_from_riva_response(response)

    assert len(words) == 1
    assert words[0]["text"] == "hello"
    assert words[0]["start"] == 0.1
    assert words[0]["end"] == 0.4
    assert words[0]["confidence"] == 0.9


def test_extract_words_falls_back_to_transcript():
    response = SimpleNamespace(
        results=[
            SimpleNamespace(
                alternatives=[
                    SimpleNamespace(
                        transcript="hello world",
                        words=[],
                    )
                ]
            )
        ]
    )

    words = _extract_words_from_riva_response(response)

    assert len(words) == 2
    assert words[0]["text"] == "hello"
    assert words[1]["text"] == "world"
