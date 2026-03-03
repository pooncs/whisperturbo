import pytest
import time
from unittest.mock import patch, MagicMock, PropertyMock


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies for TranslationPipeline."""
    with (
        patch("main.AudioInput") as mock_audio,
        patch("main.WhisperASR") as mock_whisper,
        patch("main.DiarizationHandler") as mock_diarization,
        patch("main.Fusion") as mock_fusion,
        patch("main.TranslationGUI") as mock_gui,
        patch("main.CONFIG") as mock_config,
    ):
        mock_config.SAMPLE_RATE = 16000
        mock_config.DIARIZATION_WINDOW_SIZE = 15.0

        mock_audio_instance = MagicMock()
        mock_audio.return_value = mock_audio_instance
        mock_audio_instance.vad_options = MagicMock()

        mock_whisper_instance = MagicMock()
        mock_whisper.return_value = mock_whisper_instance
        mock_whisper_instance.get_stats.return_value = {
            "last_processing_time": 0.5,
            "avg_processing_time": 0.5,
            "total_processing_time": 0.5,
            "num_transcriptions": 1,
            "model_loaded": True,
        }

        mock_diarization_instance = MagicMock()
        mock_diarization.return_value = mock_diarization_instance
        mock_diarization_instance.is_loaded = True

        mock_fusion_instance = MagicMock()
        mock_fusion.return_value = mock_fusion_instance
        mock_fusion_instance.fuse.return_value = []

        mock_gui_instance = MagicMock()
        mock_gui.return_value = mock_gui_instance

        yield {
            "audio": mock_audio,
            "whisper": mock_whisper,
            "diarization": mock_diarization,
            "fusion": mock_fusion,
            "gui": mock_gui,
            "audio_instance": mock_audio_instance,
            "whisper_instance": mock_whisper_instance,
            "diarization_instance": mock_diarization_instance,
            "fusion_instance": mock_fusion_instance,
            "gui_instance": mock_gui_instance,
        }


@pytest.fixture
def translation_pipeline(mock_dependencies):
    """Create TranslationPipeline with mocked dependencies."""
    from main import TranslationPipeline

    return TranslationPipeline(enable_gui=False, enable_diarization=False)


class TestTranslationPipelineInit:
    """Tests for TranslationPipeline initialization."""

    def test_default_initialization(self, translation_pipeline, mock_dependencies):
        """Test TranslationPipeline initializes with correct default values."""
        assert translation_pipeline.enable_gui is False
        assert translation_pipeline.enable_diarization is False
        assert translation_pipeline.gui_port == 5006
        assert translation_pipeline._running is False
        assert translation_pipeline._shutdown_event is not None

    def test_initialization_creates_components(
        self, translation_pipeline, mock_dependencies
    ):
        """Test initialization creates all components."""
        assert translation_pipeline._audio_input is not None
        assert translation_pipeline._whisper is not None
        assert translation_pipeline._fusion is not None
        assert translation_pipeline._diarization is None

    def test_initialization_with_diarization(self, mock_dependencies):
        """Test initialization creates diarization when enabled."""
        from main import TranslationPipeline

        pipeline = TranslationPipeline(enable_gui=False, enable_diarization=True)

        assert pipeline._diarization is not None

    def test_initialization_with_gui(self, mock_dependencies):
        """Test initialization creates GUI when enabled."""
        from main import TranslationPipeline

        pipeline = TranslationPipeline(enable_gui=True, enable_diarization=False)

        assert pipeline._gui is not None


class TestPipelineStart:
    """Tests for pipeline start method."""

    def test_start(self, translation_pipeline, mock_dependencies):
        """Test start initializes and starts all components."""
        translation_pipeline.start()

        assert translation_pipeline._running is True
        mock_dependencies["whisper_instance"].load_model.assert_called_once()
        mock_dependencies["audio_instance"].start.assert_called_once()

    def test_start_already_running(self, translation_pipeline, mock_dependencies):
        """Test start does nothing when already running."""
        translation_pipeline._running = True

        translation_pipeline.start()

        mock_dependencies["whisper_instance"].load_model.assert_not_called()

    def test_start_loads_diarization(self, mock_dependencies):
        """Test start loads diarization pipeline when enabled."""
        from main import TranslationPipeline

        pipeline = TranslationPipeline(enable_gui=False, enable_diarization=True)
        pipeline.start()

        mock_dependencies["diarization_instance"].load_pipeline.assert_called_once()

    def test_start_starts_gui(self, mock_dependencies):
        """Test start serves GUI when enabled."""
        from main import TranslationPipeline

        pipeline = TranslationPipeline(enable_gui=True, enable_diarization=False)
        pipeline.start()

        mock_dependencies["gui_instance"].serve.assert_called_once_with(port=5006)


class TestPipelineStop:
    """Tests for pipeline stop method."""

    def test_stop(self, translation_pipeline, mock_dependencies):
        """Test stop stops all components."""
        translation_pipeline._running = True
        translation_pipeline._process_thread = MagicMock()

        translation_pipeline.stop()

        assert translation_pipeline._running is False
        mock_dependencies["audio_instance"].stop.assert_called_once()
        mock_dependencies["whisper_instance"].unload_model.assert_called_once()

    def test_stop_not_running(self, translation_pipeline, mock_dependencies):
        """Test stop does nothing when not running."""
        translation_pipeline._running = False

        translation_pipeline.stop()

        mock_dependencies["audio_instance"].stop.assert_not_called()

    def test_stop_unloads_diarization(self, mock_dependencies):
        """Test stop unloads diarization when enabled."""
        from main import TranslationPipeline

        pipeline = TranslationPipeline(enable_gui=False, enable_diarization=True)
        pipeline._running = True
        pipeline._process_thread = MagicMock()

        pipeline.stop()

        mock_dependencies["diarization_instance"].unload_pipeline.assert_called_once()

    def test_stop_stops_gui(self, mock_dependencies):
        """Test stop stops GUI when enabled."""
        from main import TranslationPipeline

        pipeline = TranslationPipeline(enable_gui=True, enable_diarization=False)
        pipeline._running = True
        pipeline._gui = MagicMock()
        pipeline._process_thread = MagicMock()

        pipeline.stop()

        pipeline._gui.stop.assert_called_once()


class TestPipelineLifecycle:
    """Tests for pipeline lifecycle methods."""

    def test_is_running_property(self, translation_pipeline):
        """Test is_running property returns correct value."""
        assert translation_pipeline.is_running is False

        translation_pipeline._running = True

        assert translation_pipeline.is_running is True


class TestProcessAudio:
    """Tests for audio processing."""

    def test_process_audio_not_running(self, translation_pipeline, mock_dependencies):
        """Test _process_audio returns when audio input not running."""
        mock_dependencies["audio_instance"].is_running = False

        translation_pipeline._process_audio()

        mock_dependencies["whisper_instance"].transcribe.assert_not_called()

    def test_process_audio_insufficient_audio(
        self, translation_pipeline, mock_dependencies
    ):
        """Test _process_audio returns when insufficient audio."""
        mock_dependencies["audio_instance"].is_running = True
        mock_dependencies["audio_instance"].get_recent_audio.return_value = MagicMock(
            __len__=lambda self: 4000
        )

        translation_pipeline._process_audio()

        mock_dependencies["whisper_instance"].transcribe.assert_not_called()

    def test_process_audio_success(self, translation_pipeline, mock_dependencies):
        """Test _process_audio processes audio successfully."""
        mock_dependencies["audio_instance"].is_running = True
        mock_dependencies["audio_instance"].get_recent_audio.return_value = MagicMock(
            __len__=lambda self: 16000 * 5
        )
        mock_dependencies["whisper_instance"].transcribe.return_value = [
            MagicMock(start=0, end=1, text="Hello", language="en", no_speech_prob=0.1)
        ]

        translation_pipeline._process_audio()

        mock_dependencies["whisper_instance"].transcribe.assert_called_once()
        mock_dependencies["fusion_instance"].fuse.assert_called_once()

    def test_process_audio_with_diarization(self, mock_dependencies):
        """Test _process_audio includes diarization when enabled."""
        from main import TranslationPipeline

        pipeline = TranslationPipeline(enable_gui=False, enable_diarization=True)
        pipeline._running = True
        pipeline._audio_input = MagicMock()
        pipeline._audio_input.is_running = True
        pipeline._audio_input.get_recent_audio.return_value = MagicMock(
            __len__=lambda self: 16000 * 5
        )
        pipeline._audio_input.vad_options = MagicMock()
        pipeline._audio_input.get_current_timestamp.return_value = 20.0

        pipeline._whisper = MagicMock()
        pipeline._whisper.transcribe.return_value = [
            MagicMock(start=0, end=1, text="Hello", language="en", no_speech_prob=0.1)
        ]

        pipeline._diarization = MagicMock()
        pipeline._diarization.diarize_audio.return_value = []

        pipeline._fusion = MagicMock()

        pipeline._process_audio()

        pipeline._diarization.diarize_audio.assert_called_once()


class TestProcessLoop:
    """Tests for processing loop."""

    def test_process_loop_stops_on_shutdown(
        self, translation_pipeline, mock_dependencies
    ):
        """Test process_loop stops when shutdown event is set."""
        translation_pipeline._shutdown_event.set()

        translation_pipeline._process_loop()

        assert True


class TestSetupComponents:
    """Tests for component setup."""

    def test_setup_components(self, translation_pipeline, mock_dependencies):
        """Test _setup_components initializes all components."""
        mock_dependencies["audio"].assert_called()
        mock_dependencies["whisper"].assert_called()
        mock_dependencies["fusion"].assert_called()
