import time
import threading
import numpy as np
from typing import List, Optional, Generator, Any
from dataclasses import dataclass
import logging

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from .config import CONFIG


logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    language: str
    avg_logprob: float = -1.0
    no_speech_prob: float = 0.0


class WhisperASR:
    def __init__(
        self,
        model_name: str = CONFIG.WHISPER_MODEL,
        device: str = CONFIG.WHISPER_DEVICE,
        compute_type: str = CONFIG.WHISPER_COMPUTE_TYPE,
        task: str = CONFIG.WHISPER_TASK,
        language: str = CONFIG.WHISPER_LANGUAGE,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.task = task
        self.language = language

        self._model: Optional[WhisperModel] = None
        self._is_loaded = False
        self._load_lock = threading.Lock()

        self._last_transcription_time = 0.0
        self._total_processing_time = 0.0
        self._num_transcriptions = 0

    def load_model(self) -> None:
        if self._is_loaded:
            return

        with self._load_lock:
            if self._is_loaded:
                return

            logger.info(f"Loading Whisper model: {self.model_name}")
            start_time = time.time()

            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )

            elapsed = time.time() - start_time
            logger.info(f"Whisper model loaded in {elapsed:.2f}s")
            self._is_loaded = True

    def unload_model(self) -> None:
        if self._model:
            del self._model
            self._model = None
            self._is_loaded = False
            logger.info("Whisper model unloaded")

    def transcribe(
        self,
        audio: np.ndarray,
        vad_options: Optional[Any] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> List[TranscriptionSegment]:
        if not self._is_loaded:
            self.load_model()

        start_time = time.time()

        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            task=self.task,
            beam_size=beam_size,
            vad_options=vad_options,
            vad_filter=vad_filter,
            word_timestamps=False,
        )

        results = []
        for segment in segments:
            results.append(
                TranscriptionSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    language=info.language if info.language else self.language,
                    avg_logprob=segment.avg_logprob
                    if hasattr(segment, "avg_logprob")
                    else -1.0,
                    no_speech_prob=segment.no_speech_prob
                    if hasattr(segment, "no_speech_prob")
                    else 0.0,
                )
            )

        processing_time = time.time() - start_time
        self._total_processing_time += processing_time
        self._num_transcriptions += 1
        self._last_transcription_time = processing_time

        audio_duration = len(audio) / CONFIG.SAMPLE_RATE
        rtf = processing_time / audio_duration if audio_duration > 0 else 0

        logger.debug(
            f"Transcribed {audio_duration:.2f}s audio in {processing_time:.2f}s "
            f"(RTF: {rtf:.2f}x, {len(results)} segments)"
        )

        return results

    def transcribe_streaming(
        self,
        audio_generator: Generator[np.ndarray, None, None],
        vad_options: Optional[Any] = None,
    ) -> Generator[TranscriptionSegment, None, None]:
        if not self._is_loaded:
            self.load_model()

        for audio_chunk in audio_generator:
            segments = self.transcribe(audio_chunk, vad_options=vad_options)
            for segment in segments:
                yield segment

    def get_stats(self) -> dict:
        avg_processing_time = (
            self._total_processing_time / self._num_transcriptions
            if self._num_transcriptions > 0
            else 0
        )

        return {
            "last_processing_time": self._last_transcription_time,
            "avg_processing_time": avg_processing_time,
            "total_processing_time": self._total_processing_time,
            "num_transcriptions": self._num_transcriptions,
            "model_loaded": self._is_loaded,
        }

    def reset_stats(self) -> None:
        self._last_transcription_time = 0.0
        self._total_processing_time = 0.0
        self._num_transcriptions = 0

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
