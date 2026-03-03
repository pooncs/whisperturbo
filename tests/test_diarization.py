import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock


@pytest.fixture
def mock_pyannote():
    """Mock pyannote modules."""
    with (
        patch("src.diarization.Pipeline") as mock_pipeline,
        patch("src.diarization.Segment") as mock_segment,
        patch("src.diarization.Annotation") as mock_annotation,
        patch("src.diarization.torch") as mock_torch,
    ):
        mock_pipeline_instance = MagicMock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance
        mock_torch.device.return_value = MagicMock()

        mock_diar = MagicMock()
        mock_diar.itertracks.return_value = [
            (MagicMock(start=0.0, end=1.5), None, "SPEAKER_00"),
            (MagicMock(start=1.5, end=3.0), None, "SPEAKER_01"),
        ]
        mock_pipeline_instance.return_value = mock_diar

        yield {
            "pipeline": mock_pipeline,
            "pipeline_instance": mock_pipeline_instance,
            "diar": mock_diar,
            "torch": mock_torch,
        }


@pytest.fixture
def mock_config():
    """Mock CONFIG for diarization tests."""
    with patch("src.diarization.CONFIG") as mock_cfg:
        mock_cfg.SAMPLE_RATE = 16000
        mock_cfg.WHISPER_DEVICE = "cuda"
        mock_cfg.DIARIZATION_MODEL = "pyannote/speaker-diarization-1"
        yield mock_cfg


@pytest.fixture
def diarization_handler(mock_pyannote, mock_config):
    """Create DiarizationHandler instance with mocked dependencies."""
    from src.diarization import DiarizationHandler

    return DiarizationHandler(hf_token="test_token")


class TestDiarizationHandlerInit:
    """Tests for DiarizationHandler initialization."""

    def test_default_initialization(self, diarization_handler):
        """Test DiarizationHandler initializes with correct default values."""
        assert (
            diarization_handler.model_name == "pyannote/speaker-diarization-community-1"
        )
        assert diarization_handler.window_size == 15.0
        assert diarization_handler.overlap == 5.0
        assert diarization_handler.hf_token == "test_token"
        assert diarization_handler._pipeline is None
        assert diarization_handler._is_loaded is False

    def test_custom_initialization(self, mock_pyannote, mock_config):
        """Test DiarizationHandler initializes with custom parameters."""
        from src.diarization import DiarizationHandler

        handler = DiarizationHandler(
            model_name="custom/model",
            window_size=20.0,
            overlap=3.0,
            hf_token="custom_token",
        )

        assert handler.model_name == "custom/model"
        assert handler.window_size == 20.0
        assert handler.overlap == 3.0
        assert handler.hf_token == "custom_token"

    def test_uses_config_token_when_none_provided(self, mock_pyannote, mock_config):
        """Test DiarizationHandler uses CONFIG.hf_token when no token provided."""
        with patch("src.diarization.CONFIG") as mock_cfg:
            mock_cfg.hf_token = "config_token"
            mock_cfg.SAMPLE_RATE = 16000
            mock_cfg.WHISPER_DEVICE = "cuda"
            mock_cfg.DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

            from src.diarization import DiarizationHandler

            handler = DiarizationHandler()

            assert handler.hf_token == "config_token"


class TestDiarizationPipeline:
    """Tests for pipeline loading/unloading."""

    def test_load_pipeline(self, diarization_handler, mock_pyannote):
        """Test load_pipeline loads the diarization pipeline."""
        diarization_handler.load_pipeline()

        assert diarization_handler._is_loaded
        mock_pyannote["pipeline"].from_pretrained.assert_called_once_with(
            "pyannote/speaker-diarization-community-1",
            use_auth_token="test_token",
        )

    def test_load_pipeline_already_loaded(self, diarization_handler, mock_pyannote):
        """Test load_pipeline does nothing when already loaded."""
        diarization_handler._is_loaded = True

        diarization_handler.load_pipeline()

        mock_pyannote["pipeline"].from_pretrained.assert_not_called()

    def test_load_pipeline_moves_to_cuda(
        self, diarization_handler, mock_pyannote, mock_config
    ):
        """Test load_pipeline moves pipeline to CUDA when configured."""
        diarization_handler.load_pipeline()

        mock_pyannote["torch"].device.assert_called_with("cuda")

    def test_unload_pipeline(self, diarization_handler, mock_pyannote):
        """Test unload_pipeline unloads the pipeline."""
        diarization_handler._pipeline = MagicMock()
        diarization_handler._is_loaded = True

        diarization_handler.unload_pipeline()

        assert diarization_handler._pipeline is None
        assert diarization_handler._is_loaded is False


class TestDiarizationProcessing:
    """Tests for diarization processing."""

    def test_process_window(self, diarization_handler, mock_pyannote):
        """Test _process_window returns speaker segments."""
        audio = np.random.rand(16000 * 10).astype(np.float32)

        diarization_handler.load_pipeline()

        segments = diarization_handler._process_window(audio, 0.0)

        assert len(segments) == 2

    def test_diarize_audio_short(self, diarization_handler, mock_pyannote):
        """Test diarize_audio returns empty for short audio."""
        audio = np.random.rand(16000 * 2).astype(np.float32)

        segments = diarization_handler.diarize_audio(audio, 0.0)

        assert len(segments) == 0

    def test_diarize_audio_long(self, diarization_handler, mock_pyannote):
        """Test diarize_audio processes long audio."""
        audio = np.random.rand(16000 * 10).astype(np.float32)

        diarization_handler.load_pipeline()

        segments = diarization_handler.diarize_audio(audio, 0.0)

        assert len(segments) > 0


class TestDiarizationAsync:
    """Tests for async diarization processing."""

    def test_process_async(self, diarization_handler, mock_pyannote):
        """Test process_async starts async processing."""
        audio = np.random.rand(16000 * 10).astype(np.float32)

        diarization_handler.load_pipeline()
        diarization_handler.process_async(audio, 0.0)

        # Wait for async processing to complete
        if diarization_handler._processing_thread:
            diarization_handler._processing_thread.join(timeout=5.0)

        # Verify results were processed
        results = diarization_handler.get_results(timeout=0.1)
        assert isinstance(results, list)

    def test_get_results(self, diarization_handler, mock_pyannote):
        """Test get_results returns segments from queue."""
        test_segments = [MagicMock(), MagicMock()]
        diarization_handler._result_queue.put(test_segments)

        results = diarization_handler.get_results(timeout=0.1)

        assert results == test_segments

    def test_get_results_empty(self, diarization_handler):
        """Test get_results returns empty list when queue is empty."""
        import queue

        results = diarization_handler.get_results(timeout=0.01)

        assert results == []


class TestSpeakerAtTime:
    """Tests for speaker lookup by time."""

    def test_get_speaker_at_time_found(self, diarization_handler, mock_pyannote):
        """Test get_speaker_at_time returns speaker when found."""
        from src.diarization import SpeakerSegment

        diarization_handler._result_queue.put(
            [
                SpeakerSegment(0.0, 2.0, "SPEAKER_00"),
                SpeakerSegment(2.0, 4.0, "SPEAKER_01"),
            ]
        )

        speaker = diarization_handler.get_speaker_at_time(1.0)

        assert speaker == "SPEAKER_00"

    def test_get_speaker_at_time_not_found(self, diarization_handler):
        """Test get_speaker_at_time returns None when not found."""
        diarization_handler._result_queue.put([])

        speaker = diarization_handler.get_speaker_at_time(100.0)

        assert speaker is None


class TestDiarizationStats:
    """Tests for diarization statistics."""

    def test_get_stats(self, diarization_handler):
        """Test get_stats returns correct structure."""
        stats = diarization_handler.get_stats()

        assert "pipeline_loaded" in stats
        assert "is_processing" in stats
        assert "buffer_size" in stats
        assert stats["pipeline_loaded"] is False
        assert stats["is_processing"] is False

    def test_get_stats_with_data(self, diarization_handler):
        """Test get_stats reflects current state."""
        diarization_handler._is_processing = True
        diarization_handler._audio_buffer = [np.array([1, 2, 3])]

        stats = diarization_handler.get_stats()

        assert stats["is_processing"] is True
        assert stats["buffer_size"] == 1


class TestSpeakerSegment:
    """Tests for SpeakerSegment dataclass."""

    def test_speaker_segment_creation(self):
        """Test SpeakerSegment can be created with required fields."""
        from src.diarization import SpeakerSegment

        seg = SpeakerSegment(
            start=0.0,
            end=1.5,
            speaker="SPEAKER_00",
        )

        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.speaker == "SPEAKER_00"

    def test_speaker_segment_defaults(self):
        """Test SpeakerSegment has correct default values."""
        from src.diarization import SpeakerSegment

        seg = SpeakerSegment(
            start=0.0,
            end=1.0,
            speaker="SPEAKER_00",
        )

        assert seg.confidence == 1.0


class TestDiarizationRolling:
    """Tests for rolling diarization."""

    def test_diarize_rolling_short_buffer(self, diarization_handler, mock_pyannote):
        """Test diarize_rolling returns empty for short buffer."""
        audio_input = MagicMock()
        audio_input.get_current_timestamp.return_value = 5.0

        segments = diarization_handler.diarize_rolling(audio_input)

        assert len(segments) == 0

    def test_diarize_rolling_long_buffer(
        self, diarization_handler, mock_pyannote, mock_config
    ):
        """Test diarize_rolling processes when buffer is long enough."""
        audio_input = MagicMock()
        audio_input.get_current_timestamp.return_value = 20.0
        audio_input.get_audio_window.return_value = np.random.rand(16000 * 15).astype(
            np.float32
        )

        diarization_handler.load_pipeline()

        segments = diarization_handler.diarize_rolling(audio_input)

        audio_input.get_audio_window.assert_called_once()
