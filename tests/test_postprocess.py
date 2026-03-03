import pytest
from src.postprocess import normalize_whitespace, trim_repetitions, merge_short_segments
from src.fusion import TranslatedSegment


class TestNormalizeWhitespace:
    def test_normalize_whitespace_collapse_spaces(self):
        text = "Hello    world"
        result = normalize_whitespace(text)
        assert result == "Hello world"

    def test_normalize_whitespace_trim(self):
        text = "   Hello world   "
        result = normalize_whitespace(text)
        assert result == "Hello world"

    def test_normalize_whitespace_tabs_newlines(self):
        text = "Hello\n\tworld"
        result = normalize_whitespace(text)
        assert result == "Hello world"

    def test_normalize_whitespace_empty(self):
        text = ""
        result = normalize_whitespace(text)
        assert result == ""


class TestTrimRepetitions:
    def test_trim_repetitions_basic(self):
        text = "the the the cat sat"
        result = trim_repetitions(text)
        assert result == "the cat sat"

    def test_trim_repetitions_no_change(self):
        text = "the quick brown fox"
        result = trim_repetitions(text)
        assert result == "the quick brown fox"

    def test_trim_repetitions_case_insensitive(self):
        text = "The THE cat"
        result = trim_repetitions(text)
        assert result == "The cat"

    def test_trim_repetitions_empty(self):
        text = ""
        result = trim_repetitions(text)
        assert result == ""

    def test_trim_repetitions_single_word(self):
        text = "word"
        result = trim_repetitions(text)
        assert result == "word"


class TestMergeShortSegments:
    def test_merge_short_segments_same_speaker(self):
        segments = [
            TranslatedSegment(
                start=0.0, end=0.5, text="Hello", speaker="SPEAKER_00", language="en"
            ),
            TranslatedSegment(
                start=0.6, end=0.8, text="world", speaker="SPEAKER_00", language="en"
            ),
        ]
        result = merge_short_segments(segments, min_duration=1.0)
        assert len(result) == 1
        assert result[0].text == "Hello world"
        assert result[0].end == 0.8

    def test_merge_short_segments_different_speaker(self):
        segments = [
            TranslatedSegment(
                start=0.0, end=0.5, text="Hello", speaker="SPEAKER_00", language="en"
            ),
            TranslatedSegment(
                start=0.6, end=0.8, text="world", speaker="SPEAKER_01", language="en"
            ),
        ]
        result = merge_short_segments(segments, min_duration=1.0)
        assert len(result) == 2

    def test_merge_short_segments_long_segment(self):
        segments = [
            TranslatedSegment(
                start=0.0, end=2.0, text="Hello", speaker="SPEAKER_00", language="en"
            ),
            TranslatedSegment(
                start=2.1, end=3.5, text="world", speaker="SPEAKER_00", language="en"
            ),
        ]
        result = merge_short_segments(segments, min_duration=1.0)
        assert len(result) == 2

    def test_merge_short_segments_empty(self):
        result = merge_short_segments([], min_duration=1.0)
        assert result == []

    def test_merge_short_segments_single(self):
        segments = [
            TranslatedSegment(
                start=0.0, end=0.5, text="Hello", speaker="SPEAKER_00", language="en"
            ),
        ]
        result = merge_short_segments(segments, min_duration=1.0)
        assert len(result) == 1
