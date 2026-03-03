import time
import threading
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
import queue

import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation

from .config import CONFIG


logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    confidence: float = 1.0


class DiarizationHandler:
    def __init__(
        self,
        model_name: str = CONFIG.DIARIZATION_MODEL,
        window_size: float = CONFIG.DIARIZATION_WINDOW_SIZE,
        overlap: float = CONFIG.DIARIZATION_OVERLAP,
        hf_token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.window_size = window_size
        self.overlap = overlap
        self.hf_token = hf_token or CONFIG.hf_token

        if not self.hf_token:
            logger.warning("HF_TOKEN not set. Diarization may fail.")

        self._pipeline: Optional[Pipeline] = None
        self._is_loaded = False
        self._load_lock = threading.Lock()

        self._result_queue: queue.Queue = queue.Queue()
        self._is_processing = False
        self._processing_thread: Optional[threading.Thread] = None

        self._audio_buffer: List[np.ndarray] = []
        self._audio_buffer_lock = threading.Lock()

        self._last_processed_end = 0.0
        self._current_speakers: Dict[str, float] = {}

    def load_pipeline(self) -> None:
        if self._is_loaded:
            return

        with self._load_lock:
            if self._is_loaded:
                return

            logger.info(f"Loading diarization pipeline: {self.model_name}")
            start_time = time.time()

            self._pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token,
            )

            if CONFIG.WHISPER_DEVICE == "cuda":
                try:
                    self._pipeline.to(torch.device("cuda"))
                except Exception as e:
                    logger.warning(f"Could not move pipeline to CUDA: {e}")

            elapsed = time.time() - start_time
            logger.info(f"Diarization pipeline loaded in {elapsed:.2f}s")
            self._is_loaded = True

    def unload_pipeline(self) -> None:
        if self._pipeline:
            del self._pipeline
            self._pipeline = None
            self._is_loaded = False
            logger.info("Diarization pipeline unloaded")

    def _process_window(
        self, audio: np.ndarray, start_time: float
    ) -> List[SpeakerSegment]:
        if not self._is_loaded:
            self.load_pipeline()

        try:
            diarization = self._pipeline(
                {"waveform": audio, "sample_rate": CONFIG.SAMPLE_RATE}
            )

            results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                results.append(
                    SpeakerSegment(
                        start=turn.start + start_time,
                        end=turn.end + start_time,
                        speaker=speaker,
                        confidence=1.0,
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Diarization error: {e}")
            return []

    def add_audio_chunk(self, audio: np.ndarray) -> None:
        with self._audio_buffer_lock:
            self._audio_buffer.append(audio)

            total_duration = (
                sum(len(a) for a in self._audio_buffer) / CONFIG.SAMPLE_RATE
            )
            while total_duration > self.window_size * 2:
                removed = self._audio_buffer.pop(0)
                total_duration -= len(removed) / CONFIG.SAMPLE_RATE

    def process_async(self, audio: np.ndarray, timestamp: float) -> None:
        if not self._is_loaded:
            self.load_pipeline()

        if self._is_processing:
            return

        self._is_processing = True

        def _run():
            try:
                segments = self._process_window(audio, timestamp)
                self._result_queue.put(segments)
            except Exception as e:
                logger.error(f"Async diarization error: {e}")
                self._result_queue.put([])
            finally:
                self._is_processing = False

        self._processing_thread = threading.Thread(target=_run, daemon=True)
        self._processing_thread.start()

    def get_results(self, timeout: float = 0.1) -> List[SpeakerSegment]:
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return []

    def get_speaker_at_time(self, time: float) -> Optional[str]:
        results = self.get_results(timeout=0)

        for segment in results:
            if segment.start <= time <= segment.end:
                return segment.speaker

        return None

    def diarize_audio(
        self,
        audio: np.ndarray,
        start_timestamp: float = 0.0,
    ) -> List[SpeakerSegment]:
        audio_duration = len(audio) / CONFIG.SAMPLE_RATE

        if audio_duration < 3.0:
            logger.debug("Audio too short for diarization")
            return []

        segments = self._process_window(audio, start_timestamp)
        return segments

    def diarize_rolling(
        self,
        audio_input,
        min_segment_duration: float = 1.0,
    ) -> List[SpeakerSegment]:
        current_time = audio_input.get_current_timestamp()

        if current_time < self.window_size:
            return []

        lookback_time = current_time - self.window_size
        audio = audio_input.get_audio_window(lookback_time, self.window_size)

        segments = self.diarize_audio(audio, lookback_time)

        filtered = [s for s in segments if (s.end - s.start) >= min_segment_duration]

        return filtered

    def get_stats(self) -> dict:
        return {
            "pipeline_loaded": self._is_loaded,
            "is_processing": self._is_processing,
            "buffer_size": len(self._audio_buffer),
        }

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
