import os
from dataclasses import dataclass


@dataclass
class Config:
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    DTYPE: str = "float32"
    CHUNK_DURATION: float = 0.1
    BUFFER_DURATION: float = 30.0

    WHISPER_MODEL: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
    WHISPER_DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str = "float16"
    WHISPER_TASK: str = "translate"
    WHISPER_LANGUAGE: str = "ko"

    WHISPER_BEAM_SIZE: int = 5
    WHISPER_NO_SPEECH_THRESHOLD: float = 0.6
    WHISPER_LOGPROB_THRESHOLD: float = -1.0
    WHISPER_COMPRESSION_RATIO_THRESHOLD: float = 2.4
    WHISPER_INITIAL_PROMPT: str = ""

    MIN_PROCESSING_INTERVAL: float = 2.0

    ENABLE_CONTEXT_CARRY: bool = True
    CONTEXT_MAX_LENGTH: int = 500
    CONTEXT_SEGMENT_COUNT: int = 3

    DIARIZATION_MODEL: str = "pyannote/speaker-diarization-community-1"
    DIARIZATION_WINDOW_SIZE: float = 15.0
    DIARIZATION_OVERLAP: float = 5.0

    VAD_THRESHOLD: float = 0.5
    VAD_MIN_SPEECH_DURATION: float = 0.1
    VAD_MIN_SILENCE_DURATION: float = 0.5

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
