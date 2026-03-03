import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock


@pytest.fixture
def mock_sounddevice():
    """Mock sounddevice module."""
    with patch("src.audio_input.sd") as mock_sd:
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        yield mock_sd


@pytest.fixture
def mock_vad_options():
    """Mock VadOptions from faster_whisper."""
    with patch("src.audio_input.VadOptions") as mock_vad:
        mock_vad_instance = MagicMock()
        mock_vad.return_value = mock_vad_instance
        yield mock_vad


@pytest.fixture
def audio_input(mock_sounddevice, mock_vad_options):
    """Create AudioInput instance with mocked dependencies."""
    from src.audio_input import AudioInput

    return AudioInput()


class TestAudioInputInit:
    """Tests for AudioInput initialization."""

    def test_default_initialization(self, audio_input):
        """Test AudioInput initializes with correct default values."""
        assert audio_input.sample_rate == 16000
        assert audio_input.channels == 1
        assert audio_input.dtype == "float32"
        assert audio_input.chunk_samples == 1600
        assert audio_input.buffer_samples == 480000

    def test_custom_initialization(self, mock_sounddevice, mock_vad_options):
        """Test AudioInput initializes with custom parameters."""
        from src.audio_input import AudioInput

        audio_input = AudioInput(
            sample_rate=44100,
            channels=2,
            dtype="int16",
            buffer_duration=60.0,
            chunk_duration=0.05,
        )

        assert audio_input.sample_rate == 44100
        assert audio_input.channels == 2
        assert audio_input.dtype == "int16"
        assert audio_input.chunk_samples == 2205
        assert audio_input.buffer_samples == 2646000

    def test_buffer_initialization(self, audio_input):
        """Test audio buffer is initialized correctly."""
        assert isinstance(audio_input._buffer, np.ndarray)
        assert audio_input._buffer.dtype == np.float32
        assert len(audio_input._buffer) == audio_input.buffer_samples
        assert audio_input._buffer_pos == 0
        assert audio_input._buffer_filled == 0


class TestAudioInputBufferProperties:
    """Tests for AudioInput buffer properties."""

    def test_get_buffer_empty(self, audio_input):
        """Test get_buffer returns zeros when buffer is empty."""
        buffer = audio_input.get_buffer()

        assert isinstance(buffer, np.ndarray)
        assert len(buffer) == 0

    def test_get_buffer_partial(self, audio_input):
        """Test get_buffer returns partial buffer when partially filled."""
        with audio_input._buffer_lock:
            audio_input._buffer_filled = 1000
            audio_input._buffer[:1000] = np.random.rand(1000).astype(np.float32)

        buffer = audio_input.get_buffer()

        assert len(buffer) == 1000

    def test_get_buffer_full(self, audio_input):
        """Test get_buffer returns full buffer when filled."""
        with audio_input._buffer_lock:
            audio_input._buffer_filled = audio_input.buffer_samples

        buffer = audio_input.get_buffer()

        assert len(buffer) == audio_input.buffer_samples


class TestVadOptions:
    """Tests for VAD configuration."""

    def test_vad_options_property(self, audio_input, mock_vad_options):
        """Test vad_options property returns VadOptions instance."""
        from src.audio_input import VadOptions

        vad_opts = audio_input.vad_options

        assert vad_opts is not None
        mock_vad_options.assert_called_once()
        call_kwargs = mock_vad_options.call_args[1]
        assert call_kwargs["threshold"] == 0.5
        assert call_kwargs["min_speech_duration_secs"] == 0.1
        assert call_kwargs["min_silence_duration_secs"] == 0.5


class TestGetRecentAudio:
    """Tests for get_recent_audio method."""

    def test_get_recent_audio_less_than_duration(self, audio_input):
        """Test get_recent_audio when buffer has less data than requested."""
        with audio_input._buffer_lock:
            audio_input._buffer_filled = 1000
            audio_input._buffer[:1000] = np.arange(1000, dtype=np.float32)

        result = audio_input.get_recent_audio(5.0)

        assert len(result) == 1000

    def test_get_recent_audio_more_than_duration(self, audio_input):
        """Test get_recent_audio when buffer has more data than requested."""
        samples_needed = int(5.0 * audio_input.sample_rate)

        with audio_input._buffer_lock:
            audio_input._buffer_filled = audio_input.buffer_samples
            audio_input._buffer_pos = 200000
            audio_input._buffer[:] = np.arange(
                audio_input.buffer_samples, dtype=np.float32
            )

        result = audio_input.get_recent_audio(5.0)

        assert len(result) == samples_needed

    def test_get_recent_audio_wraparound(self, audio_input):
        """Test get_recent_audio handles buffer wraparound."""
        buffer_size = audio_input.buffer_samples

        with audio_input._buffer_lock:
            audio_input._buffer_filled = buffer_size
            audio_input._buffer_pos = 500
            audio_input._buffer[:] = np.arange(buffer_size, dtype=np.float32)

        result = audio_input.get_recent_audio(5.0)

        expected_samples = int(5.0 * audio_input.sample_rate)
        assert len(result) == expected_samples


class TestGetAudioWindow:
    """Tests for get_audio_window method."""

    def test_get_audio_window_basic(self, audio_input):
        """Test get_audio_window returns correct audio slice."""
        with audio_input._buffer_lock:
            audio_input._buffer_filled = audio_input.buffer_samples
            audio_input._buffer[:] = np.arange(
                audio_input.buffer_samples, dtype=np.float32
            )

        result = audio_input.get_audio_window(1.0, 2.0)

        expected_samples = int(2.0 * audio_input.sample_rate)
        assert len(result) == expected_samples

    def test_get_audio_window_start_beyond_buffer(self, audio_input):
        """Test get_audio_window when start is beyond buffer."""
        with audio_input._buffer_lock:
            audio_input._buffer_filled = 10000

        result = audio_input.get_audio_window(100.0, 2.0)

        assert np.all(result == 0)

    def test_get_audio_window_wraparound(self, audio_input):
        """Test get_audio_window handles buffer wraparound."""
        buffer_size = audio_input.buffer_samples

        with audio_input._buffer_lock:
            audio_input._buffer_filled = buffer_size
            audio_input._buffer_pos = 1000
            audio_input._buffer[:] = np.arange(buffer_size, dtype=np.float32)

        start_seconds = buffer_size / audio_input.sample_rate - 1.0
        result = audio_input.get_audio_window(start_seconds, 2.0)

        expected_samples = int(2.0 * audio_input.sample_rate)
        assert len(result) == expected_samples


class TestTimestamp:
    """Tests for timestamp calculation."""

    def test_get_current_timestamp(self, audio_input):
        """Test get_current_timestamp calculates correctly."""
        with audio_input._buffer_lock:
            audio_input._buffer_filled = 16000

        timestamp = audio_input.get_current_timestamp()

        assert timestamp == 1.0

    def test_get_current_timestamp_empty_buffer(self, audio_input):
        """Test get_current_timestamp returns 0 for empty buffer."""
        timestamp = audio_input.get_current_timestamp()

        assert timestamp == 0.0


class TestAudioInputLifecycle:
    """Tests for AudioInput start/stop lifecycle."""

    def test_start(self, audio_input, mock_sounddevice):
        """Test start initializes and starts the audio stream."""
        audio_input.start()

        assert audio_input._is_running
        mock_sounddevice.InputStream.assert_called_once()
        mock_sounddevice.InputStream.return_value.start.assert_called_once()

    def test_start_already_running(self, audio_input, mock_sounddevice):
        """Test start does nothing when already running."""
        audio_input._is_running = True

        audio_input.start()

        mock_sounddevice.InputStream.assert_not_called()

    def test_stop(self, audio_input, mock_sounddevice):
        """Test stop stops and closes the audio stream."""
        audio_input._is_running = True
        mock_stream = MagicMock()
        audio_input._stream = mock_stream

        audio_input.stop()

        assert not audio_input._is_running
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert audio_input._stream is None

    def test_stop_not_running(self, audio_input):
        """Test stop does nothing when not running."""
        audio_input._is_running = False

        audio_input.stop()

        assert audio_input._stream is None


class TestCallback:
    """Tests for audio callback functionality."""

    def test_set_callback(self, audio_input):
        """Test set_callback stores the callback."""
        callback = MagicMock()

        audio_input.set_callback(callback)

        assert audio_input._callback == callback

    def test_audio_callback_with_callback(self, audio_input, mock_sounddevice):
        """Test audio callback triggers user callback."""
        callback = MagicMock()
        audio_input.set_callback(callback)

        indata = np.random.rand(1600, 1).astype(np.float32)

        audio_input._audio_callback(indata, 1600, None, None)

        callback.assert_called_once()

    def test_audio_callback_status_warning(self, audio_input, mock_sounddevice, caplog):
        """Test audio callback logs status warnings."""
        import logging

        with caplog.at_level(logging.WARNING):
            indata = np.random.rand(1600, 1).astype(np.float32)
            audio_input._audio_callback(indata, 1600, None, "input overflow")

        assert "input overflow" in caplog.text
