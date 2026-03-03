import os
from unittest.mock import patch


class TestConfig:
    """Tests for Config class."""

    def test_default_values(self):
        """Test that Config has correct default values."""
        from src.config import Config

        config = Config()

        assert config.SAMPLE_RATE == 16000
        assert config.CHANNELS == 1
        assert config.DTYPE == "float32"
        assert config.CHUNK_DURATION == 0.1
        assert config.BUFFER_DURATION == 30.0

        assert config.WHISPER_MODEL == "deepdml/faster-whisper-large-v3-turbo-ct2"
        assert config.WHISPER_DEVICE == "cuda"
        assert config.WHISPER_COMPUTE_TYPE == "float16"
        assert config.WHISPER_TASK == "translate"
        assert config.WHISPER_LANGUAGE == "ko"

        assert config.WHISPER_BEAM_SIZE == 5
        assert config.WHISPER_NO_SPEECH_THRESHOLD == 0.6
        assert config.WHISPER_LOGPROB_THRESHOLD == -1.0
        assert config.WHISPER_COMPRESSION_RATIO_THRESHOLD == 2.4
        assert config.WHISPER_INITIAL_PROMPT == ""

        assert config.DIARIZATION_MODEL == "pyannote/speaker-diarization-community-1"
        assert config.DIARIZATION_WINDOW_SIZE == 15.0
        assert config.DIARIZATION_OVERLAP == 5.0

        assert config.VAD_THRESHOLD == 0.5
        assert config.VAD_MIN_SPEECH_DURATION == 0.1
        assert config.VAD_MIN_SILENCE_DURATION == 0.5

        assert config.GUI_REFRESH_RATE == 100
        assert config.GUI_MAX_ROWS == 100

        assert config.EXPORT_FORMATS == ("csv", "jsonl", "srt")

    def test_hf_token_property(self):
        """Test hf_token property returns environment variable or empty string."""
        from src.config import Config

        config = Config()

        with patch.dict(os.environ, {"HF_TOKEN": "test_token_123"}):
            assert config.hf_token == "test_token_123"

        with patch.dict(os.environ, {}, clear=True):
            assert config.hf_token == ""

        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
            assert config.hf_token == ""

    def test_buffer_size_property(self):
        """Test buffer_size property calculates correctly."""
        from src.config import Config

        config = Config()

        assert config.buffer_size == int(16000 * 30.0)
        assert config.buffer_size == 480000

        config.SAMPLE_RATE = 8000
        config.BUFFER_DURATION = 10.0
        assert config.buffer_size == 80000

    def test_chunk_samples_property(self):
        """Test chunk_samples property calculates correctly."""
        from src.config import Config

        config = Config()

        assert config.chunk_samples == int(16000 * 0.1)
        assert config.chunk_samples == 1600

        config.SAMPLE_RATE = 44100
        config.CHUNK_DURATION = 0.05
        assert config.chunk_samples == 2205

    def test_model_names(self):
        """Test model name configurations."""
        from src.config import Config

        config = Config()

        assert "whisper" in config.WHISPER_MODEL.lower()
        assert "pyannote" in config.DIARIZATION_MODEL.lower()
        assert config.WHISPER_TASK in ["transcribe", "translate"]

    def test_config_singleton(self):
        """Test that CONFIG is an instance of Config and is accessible."""
        from src.config import CONFIG, Config

        assert isinstance(CONFIG, Config)
        # Verify CONFIG has expected attributes
        assert hasattr(CONFIG, "SAMPLE_RATE")
        assert hasattr(CONFIG, "WHISPER_MODEL")
        assert CONFIG.SAMPLE_RATE == 16000
