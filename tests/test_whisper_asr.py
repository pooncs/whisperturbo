from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_faster_whisper():
    """Mock faster_whisper module."""
    import time

    with (
        patch("src.whisper_asr.WhisperModel") as mock_model,
    ):
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        mock_info = MagicMock()
        mock_info.language = "ko"

        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 1.5
        mock_seg.text = " Hello world "
        mock_seg.avg_logprob = -0.5
        mock_seg.no_speech_prob = 0.1

        # Add side effect to simulate processing time
        def mock_transcribe(*args, **kwargs):
            time.sleep(0.01)  # Simulate 10ms processing
            return ([mock_seg], mock_info)

        mock_model_instance.transcribe.side_effect = mock_transcribe

        yield {
            "model": mock_model,
            "model_instance": mock_model_instance,
            "segment": mock_seg,
            "info": mock_info,
        }


@pytest.fixture
def whisper_asr(mock_faster_whisper):
    """Create WhisperASR instance with mocked dependencies."""
    from src.whisper_asr import WhisperASR

    return WhisperASR()


class TestWhisperASRInit:
    """Tests for WhisperASR initialization."""

    def test_default_initialization(self, whisper_asr):
        """Test WhisperASR initializes with correct default values."""
        assert whisper_asr.model_name == "deepdml/faster-whisper-large-v3-turbo-ct2"
        assert whisper_asr.device == "cuda"
        assert whisper_asr.compute_type == "float16"
        assert whisper_asr.task == "translate"
        assert whisper_asr.language == "ko"
        assert whisper_asr._model is None
        assert whisper_asr._is_loaded is False

    def test_custom_initialization(self, mock_faster_whisper):
        """Test WhisperASR initializes with custom parameters."""
        from src.whisper_asr import WhisperASR

        asr = WhisperASR(
            model_name="custom/model",
            device="cpu",
            compute_type="int8",
            task="transcribe",
            language="en",
        )

        assert asr.model_name == "custom/model"
        assert asr.device == "cpu"
        assert asr.compute_type == "int8"
        assert asr.task == "transcribe"
        assert asr.language == "en"


class TestWhisperModelLoading:
    """Tests for Whisper model loading/unloading."""

    def test_load_model(self, whisper_asr, mock_faster_whisper):
        """Test load_model loads the Whisper model."""
        whisper_asr.load_model()

        assert whisper_asr._is_loaded
        mock_faster_whisper["model"].assert_called_once_with(
            "deepdml/faster-whisper-large-v3-turbo-ct2",
            device="cuda",
            compute_type="float16",
        )

    def test_load_model_already_loaded(self, whisper_asr, mock_faster_whisper):
        """Test load_model does nothing when already loaded."""
        whisper_asr._is_loaded = True

        whisper_asr.load_model()

        mock_faster_whisper["model"].assert_not_called()

    def test_unload_model(self, whisper_asr, mock_faster_whisper):
        """Test unload_model unloads the model."""
        whisper_asr._model = MagicMock()
        whisper_asr._is_loaded = True

        whisper_asr.unload_model()

        assert whisper_asr._model is None
        assert whisper_asr._is_loaded is False


class TestWhisperTranscribe:
    """Tests for transcription functionality."""

    def test_transcribe(self, whisper_asr, mock_faster_whisper):
        """Test transcribe returns correct segments."""
        audio = np.random.rand(16000).astype(np.float32)

        segments = whisper_asr.transcribe(audio)

        assert len(segments) == 1
        assert segments[0].start == 0.0
        assert segments[0].end == 1.5
        assert segments[0].text == "Hello world"
        assert segments[0].language == "ko"

    def test_transcribe_auto_loads_model(self, whisper_asr, mock_faster_whisper):
        """Test transcribe automatically loads model if not loaded."""
        audio = np.random.rand(16000).astype(np.float32)

        assert not whisper_asr._is_loaded

        whisper_asr.transcribe(audio)

        assert whisper_asr._is_loaded

    def test_transcribe_with_vad_options(self, whisper_asr, mock_faster_whisper):
        """Test transcribe passes vad_options correctly."""
        audio = np.random.rand(16000).astype(np.float32)
        vad_opts = MagicMock()

        whisper_asr.transcribe(audio, vad_options=vad_opts)

        mock_faster_whisper["model_instance"].transcribe.assert_called_once()
        call_kwargs = mock_faster_whisper["model_instance"].transcribe.call_args[1]
        assert call_kwargs["vad_options"] == vad_opts

    def test_transcribe_updates_stats(self, whisper_asr, mock_faster_whisper):
        """Test transcribe updates processing statistics."""
        audio = np.random.rand(16000).astype(np.float32)

        whisper_asr.transcribe(audio)

        stats = whisper_asr.get_stats()
        assert stats["last_processing_time"] > 0
        assert stats["num_transcriptions"] == 1
        assert stats["total_processing_time"] > 0

    def test_transcribe_passes_config_thresholds(self, whisper_asr, mock_faster_whisper):
        """Test transcribe passes hallucination guard thresholds from config."""
        from src.config import CONFIG

        audio = np.random.rand(16000).astype(np.float32)

        whisper_asr.transcribe(audio)

        mock_faster_whisper["model_instance"].transcribe.assert_called_once()
        call_kwargs = mock_faster_whisper["model_instance"].transcribe.call_args[1]
        assert call_kwargs["no_speech_threshold"] == CONFIG.WHISPER_NO_SPEECH_THRESHOLD
        assert call_kwargs["log_prob_threshold"] == CONFIG.WHISPER_LOGPROB_THRESHOLD
        assert (
            call_kwargs["compression_ratio_threshold"] == CONFIG.WHISPER_COMPRESSION_RATIO_THRESHOLD
        )
        assert call_kwargs["beam_size"] == CONFIG.WHISPER_BEAM_SIZE

    def test_transcribe_with_context_carry(self, whisper_asr, mock_faster_whisper):
        """Test transcribe passes initial_prompt when context carry is enabled."""
        audio = np.random.rand(16000).astype(np.float32)
        initial_prompt = "previous text"

        whisper_asr.transcribe(
            audio, condition_on_previous_text=True, initial_prompt=initial_prompt
        )

        call_kwargs = mock_faster_whisper["model_instance"].transcribe.call_args[1]
        assert call_kwargs["initial_prompt"] == initial_prompt

    def test_transcribe_without_context_carry(self, whisper_asr, mock_faster_whisper):
        """Test transcribe does not pass initial_prompt when disabled."""
        audio = np.random.rand(16000).astype(np.float32)

        whisper_asr.transcribe(audio, condition_on_previous_text=False)

        call_kwargs = mock_faster_whisper["model_instance"].transcribe.call_args[1]
        assert "initial_prompt" not in call_kwargs


class TestWhisperStats:
    """Tests for statistics functionality."""

    def test_get_stats_initial(self, whisper_asr):
        """Test get_stats returns correct initial values."""
        stats = whisper_asr.get_stats()

        assert stats["last_processing_time"] == 0.0
        assert stats["avg_processing_time"] == 0.0
        assert stats["total_processing_time"] == 0.0
        assert stats["num_transcriptions"] == 0
        assert stats["model_loaded"] is False

    def test_get_stats_after_transcription(self, whisper_asr, mock_faster_whisper):
        """Test get_stats returns updated values after transcription."""
        audio = np.random.rand(32000).astype(np.float32)

        whisper_asr.transcribe(audio)

        stats = whisper_asr.get_stats()

        assert stats["num_transcriptions"] == 1
        assert stats["last_processing_time"] > 0
        assert stats["model_loaded"] is True

    def test_reset_stats(self, whisper_asr, mock_faster_whisper):
        """Test reset_stats clears all statistics."""
        audio = np.random.rand(16000).astype(np.float32)
        whisper_asr.transcribe(audio)

        whisper_asr.reset_stats()

        stats = whisper_asr.get_stats()
        assert stats["last_processing_time"] == 0.0
        assert stats["total_processing_time"] == 0.0
        assert stats["num_transcriptions"] == 0


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment dataclass."""

    def test_transcription_segment_creation(self):
        """Test TranscriptionSegment can be created with required fields."""
        from src.whisper_asr import TranscriptionSegment

        seg = TranscriptionSegment(
            start=0.0,
            end=1.5,
            text="Hello",
            language="en",
        )

        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.text == "Hello"
        assert seg.language == "en"

    def test_transcription_segment_defaults(self):
        """Test TranscriptionSegment has correct default values."""
        from src.whisper_asr import TranscriptionSegment

        seg = TranscriptionSegment(
            start=0.0,
            end=1.0,
            text="Test",
            language="en",
        )

        assert seg.avg_logprob == -1.0
        assert seg.no_speech_prob == 0.0


class TestWhisperProperties:
    """Tests for WhisperASR properties."""

    def test_is_loaded_property(self, whisper_asr):
        """Test is_loaded property returns correct value."""
        assert whisper_asr.is_loaded is False

        whisper_asr._is_loaded = True

        assert whisper_asr.is_loaded is True


class TestWhisperStreaming:
    """Tests for streaming transcription."""

    def test_transcribe_streaming(self, whisper_asr, mock_faster_whisper):
        """Test transcribe_streaming yields segments from audio chunks."""

        def audio_generator():
            for _ in range(3):
                yield np.random.rand(16000).astype(np.float32)

        mock_faster_whisper["model_instance"].transcribe.return_value = (
            [MagicMock(start=0, end=1, text="chunk", avg_logprob=-0.5, no_speech_prob=0.1)],
            MagicMock(language="ko"),
        )

        segments = list(whisper_asr.transcribe_streaming(audio_generator()))

        assert len(segments) > 0


class TestWhisperTranscribeVADChunks:
    """Tests for transcribe_vad_chunks method."""

    def test_transcribe_vad_chunks_with_vad_speech_timestamps(
        self, whisper_asr, mock_faster_whisper
    ):
        """Test transcribe_vad_chunks uses VAD and passes correct parameters."""
        from unittest.mock import patch

        audio = np.random.rand(16000 * 5).astype(np.float32)

        with patch("src.whisper_asr.get_speech_timestamps") as mock_vad:
            mock_vad.return_value = [
                {"start": 0, "end": 1500},
            ]

            def mock_transcribe_with_vad(*args, **kwargs):
                return (
                    [
                        MagicMock(
                            start=0.0,
                            end=1.5,
                            text="Hello world",
                            avg_logprob=-0.5,
                            no_speech_prob=0.1,
                        )
                    ],
                    MagicMock(language="ko"),
                )

            mock_faster_whisper["model_instance"].transcribe.side_effect = mock_transcribe_with_vad

            with patch.object(
                whisper_asr, "_build_initial_prompt", return_value="previous context"
            ):
                segments = whisper_asr.transcribe_vad_chunks(audio)

            call_kwargs = mock_faster_whisper["model_instance"].transcribe.call_args[1]
            assert call_kwargs["vad_filter"] is False

    def test_transcribe_vad_chunks_returns_segments(self, whisper_asr, mock_faster_whisper):
        """Test transcribe_vad_chunks returns TranscriptionSegment objects."""
        from unittest.mock import patch

        audio = np.random.rand(16000 * 5).astype(np.float32)

        with patch("src.whisper_asr.get_speech_timestamps") as mock_vad:
            mock_vad.return_value = [
                {"start": 0, "end": 1500},
            ]

            def mock_transcribe_with_vad(*args, **kwargs):
                return (
                    [
                        MagicMock(
                            start=0.0,
                            end=1.5,
                            text="Test segment",
                            avg_logprob=-0.5,
                            no_speech_prob=0.1,
                        )
                    ],
                    MagicMock(language="ko"),
                )

            mock_faster_whisper["model_instance"].transcribe.side_effect = mock_transcribe_with_vad

            with patch.object(whisper_asr, "_build_initial_prompt", return_value=None):
                segments = whisper_asr.transcribe_vad_chunks(audio)

        assert len(segments) == 1
        assert segments[0].text == "Test segment"
        assert segments[0].start == 0.0
        assert segments[0].end == 1.5
