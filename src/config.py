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
    
    DIARIZATION_MODEL: str = "pyannote/speaker-diarization-community-1"
    DIARIZATION_WINDOW_SIZE: float = 15.0
    DIARIZATION_OVERLAP: float = 5.0
    
    VAD_THRESHOLD: float = 0.5
    VAD_MIN_SPEECH_DURATION: float = 0.1
    VAD_MIN_SILENCE_DURATION: float = 0.5
    
    GUI_REFRESH_RATE: int = 100
    GUI_MAX_ROWS: int = 100
    
    EXPORT_FORMATS: tuple = ("csv", "jsonl", "srt")
    
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
