from src.inference.parakeet_infer import _extract_words_from_hypothesis


def test_extract_words_from_text_only_hypothesis():
    words = _extract_words_from_hypothesis({"text": "hello world"})

    assert len(words) == 2
    assert words[0]["text"] == "hello"
    assert words[0]["start"] == 0.0
    assert words[1]["text"] == "world"


def test_extract_words_from_word_timestamps():
    words = _extract_words_from_hypothesis(
        {
            "text": "hello",
            "word_timestamps": [
                {"word": "hello", "start_time": 0.1, "end_time": 0.4}
            ],
        }
    )

    assert len(words) == 1
    assert words[0]["text"] == "hello"
    assert words[0]["start"] == 0.1
    assert words[0]["end"] == 0.4
