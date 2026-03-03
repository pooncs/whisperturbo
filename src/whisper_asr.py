import logging
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps

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
    audio_time_start: Optional[float] = None
    audio_time_end: Optional[float] = None


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

        self._is_processing = False
        self._processing_lock = threading.Lock()

        self._last_transcription_time = 0.0
        self._last_finish_time = 0.0
        self._total_processing_time = 0.0
        self._num_transcriptions = 0

        self._previous_text: list[str] = []

    def add_context_text(self, text: str) -> None:
        if not text:
            return
        self._previous_text.append(text)
        max_chars = CONFIG.CONTEXT_MAX_LENGTH
        while len(self._get_context_string()) > max_chars and len(self._previous_text) > 1:
            self._previous_text.pop(0)

    def _get_context_string(self) -> str:
        segment_count = CONFIG.CONTEXT_SEGMENT_COUNT
        context_parts = self._previous_text[-segment_count:] if self._previous_text else []
        return " ".join(context_parts)

    def _build_initial_prompt(self) -> Optional[str]:
        if not CONFIG.ENABLE_CONTEXT_CARRY:
            return None
        return self._get_context_string() or None

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
        beam_size: int = CONFIG.WHISPER_BEAM_SIZE,
        vad_filter: bool = True,
        window_start_time: Optional[float] = None,
        condition_on_previous_text: bool = False,
        initial_prompt: Optional[str] = None,
    ) -> list[TranscriptionSegment]:
        if not self._is_loaded:
            self.load_model()

        start_time = time.time()

        transcribe_kwargs: dict = {
            "language": self.language,
            "task": self.task,
            "beam_size": beam_size,
            "vad_options": vad_options,
            "vad_filter": vad_filter,
            "word_timestamps": False,
            "no_speech_threshold": CONFIG.WHISPER_NO_SPEECH_THRESHOLD,
            "log_prob_threshold": CONFIG.WHISPER_LOGPROB_THRESHOLD,
            "compression_ratio_threshold": CONFIG.WHISPER_COMPRESSION_RATIO_THRESHOLD,
        }

        if condition_on_previous_text and initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt

        segments, info = self._model.transcribe(audio, **transcribe_kwargs)

        results = []
        for segment in segments:
            seg_start = segment.start
            seg_end = segment.end

            if window_start_time is not None:
                seg_start = segment.start + window_start_time
                seg_end = segment.end + window_start_time

            results.append(
                TranscriptionSegment(
                    start=seg_start,
                    end=seg_end,
                    text=segment.text.strip(),
                    language=info.language if info.language else self.language,
                    avg_logprob=(segment.avg_logprob if hasattr(segment, "avg_logprob") else -1.0),
                    no_speech_prob=(
                        segment.no_speech_prob if hasattr(segment, "no_speech_prob") else 0.0
                    ),
                    audio_time_start=(seg_start if window_start_time is not None else None),
                    audio_time_end=seg_end if window_start_time is not None else None,
                )
            )

        processing_time = time.time() - start_time
        self._total_processing_time += processing_time
        self._num_transcriptions += 1
        self._last_transcription_time = processing_time
        self._last_finish_time = time.time()

        audio_duration = len(audio) / CONFIG.SAMPLE_RATE
        rtf = processing_time / audio_duration if audio_duration > 0 else 0

        logger.debug(
            f"Transcribed {audio_duration:.2f}s audio in {processing_time:.2f}s "
            f"(RTF: {rtf:.2f}x, {len(results)} segments)"
        )

        return results

    def transcribe_vad_chunks(
        self,
        audio: np.ndarray,
        beam_size: int = CONFIG.WHISPER_BEAM_SIZE,
        window_start_time: Optional[float] = None,
    ) -> list[TranscriptionSegment]:
        if not self._is_loaded:
            self.load_model()

        with self._processing_lock:
            self._is_processing = True
            try:
                return self._transcribe_vad_chunks_internal(audio, beam_size, window_start_time)
            finally:
                self._is_processing = False

    def _transcribe_vad_chunks_internal(
        self,
        audio: np.ndarray,
        beam_size: int = CONFIG.WHISPER_BEAM_SIZE,
        window_start_time: Optional[float] = None,
    ) -> list[TranscriptionSegment]:
        initial_prompt = self._build_initial_prompt()
        use_context = CONFIG.ENABLE_CONTEXT_CARRY and initial_prompt

        # Get speech timestamps using VAD
        speech_timestamps = get_speech_timestamps(
            audio,
            sampling_rate=CONFIG.SAMPLE_RATE,
            threshold=CONFIG.VAD_THRESHOLD,
            min_speech_duration_ms=int(CONFIG.VAD_MIN_SPEECH_DURATION * 1000),
            min_silence_duration_ms=int(CONFIG.VAD_MIN_SILENCE_DURATION * 1000),
        )

        all_segments = []
        total_processing_time = 0.0

        for item in speech_timestamps:
            start_ms = item["start"]
            end_ms = item["end"]
            start_time = start_ms / 1000.0
            end_time = end_ms / 1000.0
            start_sample = int(start_time * CONFIG.SAMPLE_RATE)
            end_sample = int(end_time * CONFIG.SAMPLE_RATE)

            audio_chunk = audio[start_sample:end_sample]

            if len(audio_chunk) < CONFIG.SAMPLE_RATE * 0.1:  # skip very short chunks
                continue

            # Transcribe this chunk without VAD filter (already filtered)
            start_transcribe = time.time()
            transcribe_kwargs = {
                "language": self.language,
                "task": self.task,
                "beam_size": beam_size,
                "vad_filter": False,  # Already filtered
                "word_timestamps": False,
                "no_speech_threshold": CONFIG.WHISPER_NO_SPEECH_THRESHOLD,
                "log_prob_threshold": CONFIG.WHISPER_LOGPROB_THRESHOLD,
                "compression_ratio_threshold": CONFIG.WHISPER_COMPRESSION_RATIO_THRESHOLD,
            }
            if use_context and initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt

            segments, info = self._model.transcribe(audio_chunk, **transcribe_kwargs)

            processing_time = time.time() - start_transcribe
            total_processing_time += processing_time

            for segment in segments:
                seg_start = segment.start + start_time
                seg_end = segment.end + start_time

                if window_start_time is not None:
                    seg_start += window_start_time
                    seg_end += window_start_time

                all_segments.append(
                    TranscriptionSegment(
                        start=seg_start,
                        end=seg_end,
                        text=segment.text.strip(),
                        language=info.language if info.language else self.language,
                        avg_logprob=(
                            segment.avg_logprob if hasattr(segment, "avg_logprob") else -1.0
                        ),
                        no_speech_prob=(
                            segment.no_speech_prob if hasattr(segment, "no_speech_prob") else 0.0
                        ),
                        audio_time_start=(seg_start if window_start_time is not None else None),
                        audio_time_end=(seg_end if window_start_time is not None else None),
                    )
                )
        # Update stats
        self._total_processing_time += total_processing_time
        self._num_transcriptions += 1
        self._last_transcription_time = total_processing_time if all_segments else 0.0
        self._last_finish_time = time.time()

        audio_duration = len(audio) / CONFIG.SAMPLE_RATE
        rtf = total_processing_time / audio_duration if audio_duration > 0 else 0

        logger.debug(
            f"Transcribed VAD chunks {audio_duration:.2f}s audio in {total_processing_time:.2f}s "
            f"(RTF: {rtf:.2f}x, {len(all_segments)} segments)"
        )

        return all_segments

    def transcribe_streaming(
        self,
        audio_generator: Generator[np.ndarray, None, None],
        vad_options: Optional[Any] = None,
    ) -> Generator[TranscriptionSegment, None, None]:
        if not self._is_loaded:
            self.load_model()

        for audio_chunk in audio_generator:
            segments = self.transcribe(audio_chunk, vad_options=vad_options)
            yield from segments

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
        self._last_finish_time = 0.0
        self._total_processing_time = 0.0
        self._num_transcriptions = 0

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def is_busy(self) -> bool:
        with self._processing_lock:
            if self._is_processing:
                return True
            time_since_last = time.time() - self._last_finish_time
            return time_since_last < CONFIG.MIN_PROCESSING_INTERVAL
