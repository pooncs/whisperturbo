import pytest
import numpy as np
import csv
import json
import os
import tempfile
from unittest.mock import patch, MagicMock


@pytest.fixture
def fusion():
    """Create Fusion instance."""
    from src.fusion import Fusion

    return Fusion()


@pytest.fixture
def sample_asr_segments():
    """Create sample ASR segments."""
    from src.whisper_asr import TranscriptionSegment

    return [
        TranscriptionSegment(0.0, 1.5, "Hello world", "en", -0.5, 0.1),
        TranscriptionSegment(1.5, 3.0, "How are you", "en", -0.3, 0.05),
        TranscriptionSegment(3.0, 5.0, "I'm doing great", "en", -0.4, 0.08),
    ]


@pytest.fixture
def sample_speaker_segments():
    """Create sample speaker segments."""
    from src.diarization import SpeakerSegment

    return [
        SpeakerSegment(0.0, 2.0, "SPEAKER_00", 0.9),
        SpeakerSegment(2.0, 4.0, "SPEAKER_01", 0.85),
        SpeakerSegment(4.0, 6.0, "SPEAKER_00", 0.95),
    ]


class TestFusionInit:
    """Tests for Fusion initialization."""

    def test_default_initialization(self, fusion):
        """Test Fusion initializes with correct default values."""
        assert fusion.min_overlap == 0.3
        assert fusion._segments == []
        assert fusion._speaker_history == {}


class TestCalculateOverlap:
    """Tests for _calculate_overlap method."""

    def test_full_overlap(self, fusion):
        """Test _calculate_overlap returns 1.0 for full overlap."""
        from src.whisper_asr import TranscriptionSegment
        from src.diarization import SpeakerSegment

        asr_seg = TranscriptionSegment(1.0, 3.0, "test", "en")
        speaker_seg = SpeakerSegment(1.0, 3.0, "SPEAKER_00")

        overlap = fusion._calculate_overlap(asr_seg, speaker_seg)

        assert overlap == 1.0

    def test_partial_overlap(self, fusion):
        """Test _calculate_overlap returns correct partial overlap."""
        from src.whisper_asr import TranscriptionSegment
        from src.diarization import SpeakerSegment

        # ASR: 1.0-4.0 (duration 3.0), Speaker: 2.0-3.0 (duration 1.0)
        # Overlap: 2.0-3.0 (duration 1.0), ratio = 1.0/3.0 = 0.333
        asr_seg = TranscriptionSegment(1.0, 4.0, "test", "en")
        speaker_seg = SpeakerSegment(2.0, 3.0, "SPEAKER_00")

        overlap = fusion._calculate_overlap(asr_seg, speaker_seg)

        assert overlap == pytest.approx(0.333, rel=0.01)

    def test_no_overlap(self, fusion):
        """Test _calculate_overlap returns 0.0 for no overlap."""
        from src.whisper_asr import TranscriptionSegment
        from src.diarization import SpeakerSegment

        asr_seg = TranscriptionSegment(1.0, 2.0, "test", "en")
        speaker_seg = SpeakerSegment(3.0, 4.0, "SPEAKER_00")

        overlap = fusion._calculate_overlap(asr_seg, speaker_seg)

        assert overlap == 0.0


class TestGetDominantSpeaker:
    """Tests for _get_dominant_speaker method."""

    def test_get_dominant_speaker_with_overlap(self, fusion, sample_asr_segments):
        """Test _get_dominant_speaker returns speaker with highest overlap."""
        from src.diarization import SpeakerSegment

        speaker_segments = [
            SpeakerSegment(0.0, 2.0, "SPEAKER_00"),
            SpeakerSegment(1.5, 3.5, "SPEAKER_01"),
        ]

        speaker = fusion._get_dominant_speaker(sample_asr_segments[0], speaker_segments)

        assert speaker == "SPEAKER_00"

    def test_get_dominant_speaker_below_threshold(self, fusion, sample_asr_segments):
        """Test _get_dominant_speaker returns None when below threshold."""
        from src.diarization import SpeakerSegment

        speaker_segments = [
            SpeakerSegment(10.0, 12.0, "SPEAKER_00"),
        ]

        speaker = fusion._get_dominant_speaker(sample_asr_segments[0], speaker_segments)

        assert speaker is None

    def test_get_dominant_speaker_empty_segments(self, fusion, sample_asr_segments):
        """Test _get_dominant_speaker returns None for empty segments."""
        speaker = fusion._get_dominant_speaker(sample_asr_segments[0], [])

        assert speaker is None


class TestFuse:
    """Tests for fuse method."""

    def test_fuse_basic(self, fusion, sample_asr_segments, sample_speaker_segments):
        """Test fuse combines ASR and speaker segments."""
        fused = fusion.fuse(sample_asr_segments, sample_speaker_segments, 1000.0)

        assert len(fused) == 3
        # ASR 0.0-1.5 overlaps with SPEAKER_00 (0.0-2.0) 100% -> SPEAKER_00
        assert fused[0].speaker == "SPEAKER_00"
        # ASR 1.5-3.0 overlaps with SPEAKER_01 (2.0-4.0) 100% -> SPEAKER_01
        assert fused[1].speaker == "SPEAKER_01"
        # ASR 3.0-5.0 overlaps with SPEAKER_01 (2.0-4.0) and SPEAKER_00 (4.0-6.0) equally
        # First match (SPEAKER_01) is selected
        assert fused[2].speaker == "SPEAKER_01"

    def test_fuse_with_empty_asr(self, fusion):
        """Test fuse handles empty ASR segments."""
        fused = fusion.fuse([], [], 1000.0)

        assert len(fused) == 0

    def test_fuse_uses_last_known_speaker(self, fusion, sample_asr_segments):
        """Test fuse uses last known speaker when no overlap."""
        from src.diarization import SpeakerSegment

        # Speaker segment is far in the future (100-102s), no overlap with ASR (0-5s)
        speaker_segments = [
            SpeakerSegment(100.0, 102.0, "SPEAKER_99"),
        ]

        fused = fusion.fuse(sample_asr_segments, speaker_segments, 1000.0)

        # No overlap, so speakers should be None
        assert fused[0].speaker is None
        assert fused[1].speaker is None
        assert fused[2].speaker is None

    def test_fuse_skips_empty_text(self, fusion):
        """Test fuse skips segments with empty text."""
        from src.whisper_asr import TranscriptionSegment
        from src.diarization import SpeakerSegment

        asr_segments = [
            TranscriptionSegment(0.0, 1.0, "", "en"),
            TranscriptionSegment(1.0, 2.0, "Hello", "en"),
        ]

        fused = fusion.fuse(asr_segments, [], 1000.0)

        assert len(fused) == 1
        assert fused[0].text == "Hello"

    def test_fuse_stores_segments(
        self, fusion, sample_asr_segments, sample_speaker_segments
    ):
        """Test fuse stores segments internally."""
        fusion.fuse(sample_asr_segments, sample_speaker_segments, 1000.0)

        assert len(fusion.get_all_segments()) == 3


class TestGetAllSegments:
    """Tests for get_all_segments method."""

    def test_get_all_segments(
        self, fusion, sample_asr_segments, sample_speaker_segments
    ):
        """Test get_all_segments returns all stored segments."""
        fusion.fuse(sample_asr_segments, sample_speaker_segments, 1000.0)

        segments = fusion.get_all_segments()

        assert len(segments) == 3

    def test_get_all_segments_returns_copy(self, sample_asr_segments):
        """Test get_all_segments returns a copy."""
        from src.fusion import Fusion

        fusion = Fusion()
        fusion.fuse(sample_asr_segments, [], 1000.0)

        segments = fusion.get_all_segments()
        segments.clear()

        # Original should still have 3 segments
        assert len(fusion.get_all_segments()) == 3


class TestClear:
    """Tests for clear method."""

    def test_clear(self, fusion, sample_asr_segments, sample_speaker_segments):
        """Test clear removes all segments and history."""
        fusion.fuse(sample_asr_segments, sample_speaker_segments, 1000.0)

        fusion.clear()

        assert len(fusion.get_all_segments()) == 0
        assert fusion._speaker_history == {}


class TestGetSegmentsInRange:
    """Tests for get_segments_in_range method."""

    def test_get_segments_in_range(
        self, fusion, sample_asr_segments, sample_speaker_segments
    ):
        """Test get_segments_in_range filters correctly."""
        fusion.fuse(sample_asr_segments, sample_speaker_segments, 1000.0)

        # Segments: [0.0-1.5, 1.5-3.0, 3.0-5.0]
        # Range 1.5-3.0 includes only [1.5-3.0] (exact match)
        segments = fusion.get_segments_in_range(1.5, 3.0)

        assert len(segments) == 1
        assert segments[0].start == 1.5
        assert segments[0].end == 3.0

    def test_get_segments_in_range_no_match(
        self, fusion, sample_asr_segments, sample_speaker_segments
    ):
        """Test get_segments_in_range returns empty when no match."""
        fusion.fuse(sample_asr_segments, sample_speaker_segments, 1000.0)

        segments = fusion.get_segments_in_range(100.0, 200.0)

        assert len(segments) == 0


class TestExportCSV:
    """Tests for CSV export."""

    def test_export_csv(self, fusion, sample_asr_segments, sample_speaker_segments):
        """Test export_csv creates valid CSV file."""
        fusion.fuse(sample_asr_segments, sample_speaker_segments, 1000.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = f.name

        try:
            fusion.export_csv(filepath)

            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 3
            assert "start" in rows[0]
            assert "end" in rows[0]
            assert "speaker" in rows[0]
            assert "text" in rows[0]
        finally:
            os.unlink(filepath)


class TestExportJSONL:
    """Tests for JSONL export."""

    def test_export_jsonl(self, fusion, sample_asr_segments, sample_speaker_segments):
        """Test export_jsonl creates valid JSONL file."""
        fusion.fuse(sample_asr_segments, sample_speaker_segments, 1000.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filepath = f.name

        try:
            fusion.export_jsonl(filepath)

            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            assert len(lines) == 3
            data = json.loads(lines[0])
            assert "start" in data
            assert "end" in data
            assert "speaker" in data
            assert "text" in data
        finally:
            os.unlink(filepath)


class TestExportSRT:
    """Tests for SRT export."""

    def test_export_srt(self, fusion, sample_asr_segments, sample_speaker_segments):
        """Test export_srt creates valid SRT file."""
        fusion.fuse(sample_asr_segments, sample_speaker_segments, 1000.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            filepath = f.name

        try:
            fusion.export_srt(filepath)

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            assert "1\n" in content
            assert "00:00:00,000 --> 00:00:01,500" in content
            assert "SPEAKER_00" in content
        finally:
            os.unlink(filepath)

    def test_export_srt_timestamp_format(self, fusion):
        """Test export_srt formats timestamps correctly."""
        from src.whisper_asr import TranscriptionSegment
        from src.fusion import TranslatedSegment

        # 65.5 seconds = 1 minute 5.5 seconds = 00:01:05,500
        fusion._segments = [
            TranslatedSegment(0.0, 65.5, "test", "SPEAKER_00", "en", 1.0, 1000.0),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            filepath = f.name

        try:
            fusion.export_srt(filepath)

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # 65.5 seconds = 00:01:05,500 (1 min, 5.5 sec)
            assert "00:01:05,500" in content
        finally:
            os.unlink(filepath)


class TestExport:
    """Tests for generic export method."""

    def test_export_csv_format(self, fusion, sample_asr_segments):
        """Test export uses CSV format."""
        fusion.fuse(sample_asr_segments, [], 1000.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = f.name

        try:
            fusion.export(filepath, "csv")
            assert os.path.getsize(filepath) > 0
        finally:
            os.unlink(filepath)

    def test_export_jsonl_format(self, fusion, sample_asr_segments):
        """Test export uses JSONL format."""
        fusion.fuse(sample_asr_segments, [], 1000.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filepath = f.name

        try:
            fusion.export(filepath, "jsonl")
            assert os.path.getsize(filepath) > 0
        finally:
            os.unlink(filepath)

    def test_export_unsupported_format(self, fusion):
        """Test export raises error for unsupported format."""
        with pytest.raises(ValueError):
            fusion.export("test.txt", "txt")


class TestFusionStats:
    """Tests for get_stats method."""

    def test_get_stats_empty(self, fusion):
        """Test get_stats returns correct structure for empty fusion."""
        stats = fusion.get_stats()

        assert stats["total_segments"] == 0
        assert stats["unique_speakers"] == 0
        assert stats["speakers"] == []

    def test_get_stats_with_data(
        self, fusion, sample_asr_segments, sample_speaker_segments
    ):
        """Test get_stats returns correct stats with segments."""
        fusion.fuse(sample_asr_segments, sample_speaker_segments, 1000.0)

        stats = fusion.get_stats()

        assert stats["total_segments"] == 3
        assert stats["unique_speakers"] == 2


class TestTranslatedSegment:
    """Tests for TranslatedSegment dataclass."""

    def test_translated_segment_creation(self):
        """Test TranslatedSegment can be created with required fields."""
        from src.fusion import TranslatedSegment

        seg = TranslatedSegment(
            start=0.0,
            end=1.5,
            text="Hello",
            speaker="SPEAKER_00",
            language="en",
        )

        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.text == "Hello"
        assert seg.speaker == "SPEAKER_00"
        assert seg.language == "en"

    def test_translated_segment_defaults(self):
        """Test TranslatedSegment has correct default values."""
        from src.fusion import TranslatedSegment

        seg = TranslatedSegment(
            start=0.0,
            end=1.0,
            text="Test",
            speaker=None,
            language="en",
        )

        assert seg.confidence == 1.0
        assert seg.timestamp == 0.0
