import os
from dataclasses import dataclass, field


def _detect_device() -> str:
    """Auto-detect CUDA availability, default to CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _default_compute_type() -> str:
    """Return appropriate compute_type for the detected device."""
    if _detect_device() == "cuda":
        return "float16"
    return "int8"


@dataclass
class Config:
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    DTYPE: str = "float32"
    CHUNK_DURATION: float = 0.1
    BUFFER_DURATION: float = 30.0

    # Layered transcription models
    WHISPER_FAST_MODEL: str = "base"          # fast model for real-time (base or small)
    WHISPER_CORRECT_MODEL: str = "large-v3-turbo"  # correction model for block passes

    # Legacy single-model field (kept for backward compat, points to correct model)
    WHISPER_MODEL: str = "large-v3-turbo"

    WHISPER_DEVICE: str = field(default_factory=_detect_device)
    WHISPER_COMPUTE_TYPE: str = field(default_factory=_default_compute_type)
    WHISPER_TASK: str = "transcribe"  # transcribe in source language first, then translate
    WHISPER_LANGUAGE: str = "auto"  # auto for automatic language detection

    WHISPER_BEAM_SIZE: int = 5
    WHISPER_NO_SPEECH_THRESHOLD: float = 0.6
    WHISPER_LOGPROB_THRESHOLD: float = -1.0
    WHISPER_COMPRESSION_RATIO_THRESHOLD: float = 2.4
    WHISPER_INITIAL_PROMPT: str = ""

    MIN_PROCESSING_INTERVAL: float = 2.0

    # Correction pass settings
    CORRECTION_BLOCK_DURATION: float = 5.0  # seconds of audio per correction block

    # Summarization
    SUMMARY_ENABLED: bool = True

    ENABLE_CONTEXT_CARRY: bool = True
    CONTEXT_MAX_LENGTH: int = 500
    CONTEXT_SEGMENT_COUNT: int = 3

    # PyAnnote diarization model - community version is free but requires HF_TOKEN
    # For production, consider pyannote/speaker-diarization-3.1 (gated, requires acceptance)
    DIARIZATION_MODEL: str = "pyannote/speaker-diarization-community-1"
    DIARIZATION_WINDOW_SIZE: float = 15.0
    DIARIZATION_OVERLAP: float = 5.0

    # Pyannote segmentation model (used by diarization pipeline)
    DIARIZATION_SEGMENTATION_MODEL: str = "pyannote/segmentation-3.0"

    VAD_THRESHOLD: float = 0.3
    VAD_MIN_SPEECH_DURATION: float = 0.1
    VAD_MIN_SILENCE_DURATION: float = 0.3

    GUI_REFRESH_RATE: int = 100
    GUI_MAX_ROWS: int = 100

    EXPORT_FORMATS: tuple = ("csv", "jsonl", "srt")

    SPEAKER_COLORS: tuple = (
        "#FF0000",
        "#00FF00",
        "#0000FF",
        "#FFFF00",
        "#FF00FF",
        "#00FFFF",
        "#FFA500",
        "#800080",
        "#008080",
        "#FFC0CB",
        "#A52A2A",
        "#808080",
    )

    @property
    def hf_token(self) -> str:
        return os.environ.get("HF_TOKEN", "")

    @property
    def buffer_size(self) -> int:
        return int(self.SAMPLE_RATE * self.BUFFER_DURATION)

    @property
    def chunk_samples(self) -> int:
        return int(self.SAMPLE_RATE * self.CHUNK_DURATION)


CONFIG = Config()
