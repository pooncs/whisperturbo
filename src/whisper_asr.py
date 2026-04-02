import logging
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, load_silero_vad

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
    correction_status: str = "fast"  # "fast" or "corrected"


class WhisperASR:
    def __init__(
        self,
        model_name: str = CONFIG.WHISPER_MODEL,
        device: str = CONFIG.WHISPER_DEVICE,
        compute_type: str = CONFIG.WHISPER_COMPUTE_TYPE,
        task: str = CONFIG.WHISPER_TASK,
        language: str = CONFIG.WHISPER_LANGUAGE,
        target_language: str = "en",
        fast_model_name: str = CONFIG.WHISPER_FAST_MODEL,
        correct_model_name: str = CONFIG.WHISPER_CORRECT_MODEL,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.task = task
        self.language = language
        self.target_language = target_language

        # Dual-model names
        self.fast_model_name = fast_model_name
        self.correct_model_name = correct_model_name

        # Legacy single model (alias for correct_model for backward compat)
        self._model: Optional[WhisperModel] = None
        self._is_loaded = False

        # Dual-model instances
        self._fast_model: Optional[WhisperModel] = None
        self._fast_model_loaded = False
        self._fast_model_lock = threading.Lock()

        self._correct_model: Optional[WhisperModel] = None
        self._correct_model_loaded = False
        self._correct_model_lock = threading.Lock()

        # Shared load lock for legacy single-model path
        self._load_lock = threading.Lock()

        self._vad_model: Optional[torch.nn.Module] = None

        # Separate processing locks so fast and correct can run in parallel
        self._fast_processing_lock = threading.Lock()
        self._correct_processing_lock = threading.Lock()

        # Legacy processing lock (kept for backward compat)
        self._is_processing = False
        self._processing_lock = threading.Lock()

        self._last_transcription_time = 0.0
        self._last_finish_time = 0.0
        self._total_processing_time = 0.0
        self._num_transcriptions = 0

        self._previous_text: list[str] = []

    def set_languages(self, source_language: str, target_language: str) -> None:
        """Set source and target languages at runtime."""
        self.language = source_language
        self.target_language = target_language
        logger.info(f"Languages set: source={source_language}, target={target_language}")

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

    def _load_vad_model(self) -> torch.nn.Module:
        if self._vad_model is None:
            self._vad_model = load_silero_vad()
        return self._vad_model

    # ------------------------------------------------------------------
    # Dual-model loading / unloading
    # ------------------------------------------------------------------

    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve a model name to a loadable path. Returns the model name for HF hub lookup."""
        return model_name

    def load_fast_model(self) -> None:
        """Load the fast (small/base) model for real-time transcription."""
        if self._fast_model_loaded:
            return

        with self._fast_model_lock:
            if self._fast_model_loaded:
                return

            logger.info(f"Loading fast Whisper model: {self.fast_model_name}")
            start_time = time.time()

            self._fast_model = WhisperModel(
                self._resolve_model_path(self.fast_model_name),
                device=self.device,
                compute_type=self.compute_type,
            )

            elapsed = time.time() - start_time
            logger.info(f"Fast Whisper model loaded in {elapsed:.2f}s")
            self._fast_model_loaded = True

    def unload_fast_model(self) -> None:
        """Unload the fast model to free memory."""
        with self._fast_model_lock:
            if self._fast_model is not None:
                del self._fast_model
                self._fast_model = None
                self._fast_model_loaded = False
                logger.info("Fast Whisper model unloaded")

    def load_correct_model(self) -> None:
        """Load the correct (large) model for correction passes."""
        if self._correct_model_loaded:
            return

        with self._correct_model_lock:
            if self._correct_model_loaded:
                return

            logger.info(f"Loading correct Whisper model: {self.correct_model_name}")
            start_time = time.time()

            self._correct_model = WhisperModel(
                self._resolve_model_path(self.correct_model_name),
                device=self.device,
                compute_type=self.compute_type,
            )

            elapsed = time.time() - start_time
            logger.info(f"Correct Whisper model loaded in {elapsed:.2f}s")
            self._correct_model_loaded = True

    def unload_correct_model(self) -> None:
        """Unload the correct model to free memory."""
        with self._correct_model_lock:
            if self._correct_model is not None:
                del self._correct_model
                self._correct_model = None
                self._correct_model_loaded = False
                logger.info("Correct Whisper model unloaded")

    def load_models(self) -> None:
        """Load both fast and correct models."""
        self.load_fast_model()
        self.load_correct_model()

    def unload_models(self) -> None:
        """Unload both models."""
        self.unload_fast_model()
        self.unload_correct_model()

    # ------------------------------------------------------------------
    # Legacy single-model load/unload (backward compat)
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Legacy single-model loader. Loads the correct model (large-v3-turbo)."""
        if self._is_loaded:
            return

        with self._load_lock:
            if self._is_loaded:
                return

            logger.info(f"Loading Whisper model: {self.model_name}")
            start_time = time.time()

            model_path = self.model_name

            self._model = WhisperModel(
                model_path,
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

    # ------------------------------------------------------------------
    # Dual-model transcription methods
    # ------------------------------------------------------------------

    def transcribe_fast(
        self,
        audio: np.ndarray,
        beam_size: int = 1,
        window_start_time: Optional[float] = None,
    ) -> list[TranscriptionSegment]:
        """Transcribe audio using the fast model for real-time feedback.

        The fast model uses a smaller architecture (base/small) with minimal beam
        size for low-latency transcription on every audio chunk (~2-3s).

        Thread safety: Can run in parallel with transcribe_correct() since each
        model has its own lock.
        """
        if not self._fast_model_loaded:
            self.load_fast_model()

        with self._fast_processing_lock:
            start_time = time.time()

            source_lang = self.language if self.language != "auto" else None

            transcribe_kwargs: dict = {
                "language": source_lang,
                "task": self.task,
                "beam_size": beam_size,
                "vad_filter": False,
                "word_timestamps": False,
                "no_speech_threshold": CONFIG.WHISPER_NO_SPEECH_THRESHOLD,
                "log_prob_threshold": CONFIG.WHISPER_LOGPROB_THRESHOLD,
                "compression_ratio_threshold": CONFIG.WHISPER_COMPRESSION_RATIO_THRESHOLD,
            }

            initial_prompt = self._build_initial_prompt()
            if initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt

            try:
                segments, info = self._fast_model.transcribe(audio, **transcribe_kwargs)
            except Exception as e:
                logger.error(f"Fast model transcription failed: {e}")
                return []

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
                        avg_logprob=(
                            segment.avg_logprob if hasattr(segment, "avg_logprob") else -1.0
                        ),
                        no_speech_prob=(
                            segment.no_speech_prob if hasattr(segment, "no_speech_prob") else 0.0
                        ),
                        audio_time_start=(
                            seg_start if window_start_time is not None else None
                        ),
                        audio_time_end=(
                            seg_end if window_start_time is not None else None
                        ),
                        correction_status="fast",
                    )
                )

            processing_time = time.time() - start_time
            audio_duration = len(audio) / CONFIG.SAMPLE_RATE
            rtf = processing_time / audio_duration if audio_duration > 0 else 0

            logger.debug(
                f"Fast model transcribed {audio_duration:.2f}s in {processing_time:.2f}s "
                f"(RTF: {rtf:.2f}x, {len(results)} segments)"
            )

            return results

    def transcribe_fast_translate(
        self,
        audio: np.ndarray,
        beam_size: int = 1,
        window_start_time: Optional[float] = None,
    ) -> tuple[list[TranscriptionSegment], list[TranscriptionSegment]]:
        """Transcribe and translate using the fast model.

        Returns:
            Tuple of (source_segments, target_segments) both marked as "fast".
        """
        if not self._fast_model_loaded:
            self.load_fast_model()

        with self._fast_processing_lock:
            start_time = time.time()
            source_lang = self.language if self.language != "auto" else None

            # Step 1: Transcribe in source language
            transcribe_kwargs: dict = {
                "language": source_lang,
                "task": "transcribe",
                "beam_size": beam_size,
                "vad_filter": False,
                "word_timestamps": False,
                "no_speech_threshold": CONFIG.WHISPER_NO_SPEECH_THRESHOLD,
                "log_prob_threshold": CONFIG.WHISPER_LOGPROB_THRESHOLD,
                "compression_ratio_threshold": CONFIG.WHISPER_COMPRESSION_RATIO_THRESHOLD,
            }
            initial_prompt = self._build_initial_prompt()
            if initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt

            source_segments = []
            target_segments = []
            detected_lang = self.language

            try:
                segments, info = self._fast_model.transcribe(audio, **transcribe_kwargs)
                detected_lang = info.language if info.language else self.language
                for segment in segments:
                    seg_start = segment.start
                    seg_end = segment.end
                    if window_start_time is not None:
                        seg_start += window_start_time
                        seg_end += window_start_time
                    source_segments.append(
                        TranscriptionSegment(
                            start=seg_start,
                            end=seg_end,
                            text=segment.text.strip(),
                            language=detected_lang,
                            avg_logprob=(
                                segment.avg_logprob
                                if hasattr(segment, "avg_logprob")
                                else -1.0
                            ),
                            no_speech_prob=(
                                segment.no_speech_prob
                                if hasattr(segment, "no_speech_prob")
                                else 0.0
                            ),
                            correction_status="fast",
                        )
                    )
            except Exception as e:
                logger.error(f"Fast model transcription failed: {e}")
                return [], []

            # Step 2: Translate if needed
            if self.target_language and self.target_language != detected_lang:
                translate_kwargs = {
                    "language": self.target_language,
                    "task": "transcribe",
                    "beam_size": beam_size,
                    "vad_filter": False,
                    "word_timestamps": False,
                    "no_speech_threshold": CONFIG.WHISPER_NO_SPEECH_THRESHOLD,
                    "log_prob_threshold": CONFIG.WHISPER_LOGPROB_THRESHOLD,
                    "compression_ratio_threshold": CONFIG.WHISPER_COMPRESSION_RATIO_THRESHOLD,
                    "initial_prompt": (
                        f"Translate the following {detected_lang} speech "
                        f"to {self.target_language}."
                    ),
                }
                try:
                    trans_segments, _ = self._fast_model.transcribe(audio, **translate_kwargs)
                    for segment in trans_segments:
                        seg_start = segment.start
                        seg_end = segment.end
                        if window_start_time is not None:
                            seg_start += window_start_time
                            seg_end += window_start_time
                        target_segments.append(
                            TranscriptionSegment(
                                start=seg_start,
                                end=seg_end,
                                text=segment.text.strip(),
                                language=self.target_language,
                                avg_logprob=(
                                    segment.avg_logprob
                                    if hasattr(segment, "avg_logprob")
                                    else -1.0
                                ),
                                no_speech_prob=(
                                    segment.no_speech_prob
                                    if hasattr(segment, "no_speech_prob")
                                    else 0.0
                                ),
                                correction_status="fast",
                            )
                        )
                except Exception as e:
                    logger.error(f"Fast model translation failed: {e}")
                    for seg in source_segments:
                        target_segments.append(
                            TranscriptionSegment(
                                start=seg.start,
                                end=seg.end,
                                text=seg.text,
                                language=self.target_language,
                                avg_logprob=seg.avg_logprob,
                                no_speech_prob=seg.no_speech_prob,
                                correction_status="fast",
                            )
                        )

            processing_time = time.time() - start_time
            logger.debug(
                f"Fast model transcribe+translate in {processing_time:.2f}s "
                f"({len(source_segments)} src, {len(target_segments)} tgt)"
            )

            return source_segments, target_segments

    def transcribe_correct(
        self,
        audio: np.ndarray,
        beam_size: int = CONFIG.WHISPER_BEAM_SIZE,
        window_start_time: Optional[float] = None,
    ) -> list[TranscriptionSegment]:
        """Re-transcribe an audio block using the correct (large) model.

        This runs on larger audio blocks (~5-10s) and produces higher quality
        results that overwrite the fast model output.

        Thread safety: Uses its own lock so it can run alongside transcribe_fast().
        """
        if not self._correct_model_loaded:
            self.load_correct_model()

        with self._correct_processing_lock:
            start_time = time.time()

            source_lang = self.language if self.language != "auto" else None

            transcribe_kwargs: dict = {
                "language": source_lang,
                "task": self.task,
                "beam_size": beam_size,
                "vad_filter": True,
                "word_timestamps": False,
                "no_speech_threshold": CONFIG.WHISPER_NO_SPEECH_THRESHOLD,
                "log_prob_threshold": CONFIG.WHISPER_LOGPROB_THRESHOLD,
                "compression_ratio_threshold": CONFIG.WHISPER_COMPRESSION_RATIO_THRESHOLD,
            }

            initial_prompt = self._build_initial_prompt()
            if initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt

            try:
                segments, info = self._correct_model.transcribe(audio, **transcribe_kwargs)
            except Exception as e:
                logger.error(f"Correct model transcription failed: {e}")
                return []

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
                        avg_logprob=(
                            segment.avg_logprob if hasattr(segment, "avg_logprob") else -1.0
                        ),
                        no_speech_prob=(
                            segment.no_speech_prob if hasattr(segment, "no_speech_prob") else 0.0
                        ),
                        audio_time_start=(
                            seg_start if window_start_time is not None else None
                        ),
                        audio_time_end=(
                            seg_end if window_start_time is not None else None
                        ),
                        correction_status="corrected",
                    )
                )

            processing_time = time.time() - start_time
            audio_duration = len(audio) / CONFIG.SAMPLE_RATE
            rtf = processing_time / audio_duration if audio_duration > 0 else 0

            logger.debug(
                f"Correct model transcribed {audio_duration:.2f}s in {processing_time:.2f}s "
                f"(RTF: {rtf:.2f}x, {len(results)} segments)"
            )

            return results

    def transcribe_correct_translate(
        self,
        audio: np.ndarray,
        beam_size: int = CONFIG.WHISPER_BEAM_SIZE,
        window_start_time: Optional[float] = None,
    ) -> tuple[list[TranscriptionSegment], list[TranscriptionSegment]]:
        """Re-transcribe and translate using the correct model on a larger block.

        Returns:
            Tuple of (source_segments, target_segments) both marked as "corrected".
        """
        if not self._correct_model_loaded:
            self.load_correct_model()

        with self._correct_processing_lock:
            start_time = time.time()
            source_lang = self.language if self.language != "auto" else None

            # Step 1: Transcribe
            transcribe_kwargs: dict = {
                "language": source_lang,
                "task": "transcribe",
                "beam_size": beam_size,
                "vad_filter": True,
                "word_timestamps": False,
                "no_speech_threshold": CONFIG.WHISPER_NO_SPEECH_THRESHOLD,
                "log_prob_threshold": CONFIG.WHISPER_LOGPROB_THRESHOLD,
                "compression_ratio_threshold": CONFIG.WHISPER_COMPRESSION_RATIO_THRESHOLD,
            }
            initial_prompt = self._build_initial_prompt()
            if initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt

            source_segments = []
            target_segments = []
            detected_lang = self.language

            try:
                segments, info = self._correct_model.transcribe(audio, **transcribe_kwargs)
                detected_lang = info.language if info.language else self.language
                for segment in segments:
                    seg_start = segment.start
                    seg_end = segment.end
                    if window_start_time is not None:
                        seg_start += window_start_time
                        seg_end += window_start_time
                    source_segments.append(
                        TranscriptionSegment(
                            start=seg_start,
                            end=seg_end,
                            text=segment.text.strip(),
                            language=detected_lang,
                            avg_logprob=(
                                segment.avg_logprob
                                if hasattr(segment, "avg_logprob")
                                else -1.0
                            ),
                            no_speech_prob=(
                                segment.no_speech_prob
                                if hasattr(segment, "no_speech_prob")
                                else 0.0
                            ),
                            correction_status="corrected",
                        )
                    )
            except Exception as e:
                logger.error(f"Correct model transcription failed: {e}")
                return [], []

            # Step 2: Translate if needed
            if self.target_language and self.target_language != detected_lang:
                translate_kwargs = {
                    "language": self.target_language,
                    "task": "transcribe",
                    "beam_size": beam_size,
                    "vad_filter": True,
                    "word_timestamps": False,
                    "no_speech_threshold": CONFIG.WHISPER_NO_SPEECH_THRESHOLD,
                    "log_prob_threshold": CONFIG.WHISPER_LOGPROB_THRESHOLD,
                    "compression_ratio_threshold": CONFIG.WHISPER_COMPRESSION_RATIO_THRESHOLD,
                    "initial_prompt": (
                        f"Translate the following {detected_lang} speech "
                        f"to {self.target_language}."
                    ),
                }
                try:
                    trans_segments, _ = self._correct_model.transcribe(
                        audio, **translate_kwargs
                    )
                    for segment in trans_segments:
                        seg_start = segment.start
                        seg_end = segment.end
                        if window_start_time is not None:
                            seg_start += window_start_time
                            seg_end += window_start_time
                        target_segments.append(
                            TranscriptionSegment(
                                start=seg_start,
                                end=seg_end,
                                text=segment.text.strip(),
                                language=self.target_language,
                                avg_logprob=(
                                    segment.avg_logprob
                                    if hasattr(segment, "avg_logprob")
                                    else -1.0
                                ),
                                no_speech_prob=(
                                    segment.no_speech_prob
                                    if hasattr(segment, "no_speech_prob")
                                    else 0.0
                                ),
                                correction_status="corrected",
                            )
                        )
                except Exception as e:
                    logger.error(f"Correct model translation failed: {e}")
                    for seg in source_segments:
                        target_segments.append(
                            TranscriptionSegment(
                                start=seg.start,
                                end=seg.end,
                                text=seg.text,
                                language=self.target_language,
                                avg_logprob=seg.avg_logprob,
                                no_speech_prob=seg.no_speech_prob,
                                correction_status="corrected",
                            )
                        )

            processing_time = time.time() - start_time
            logger.debug(
                f"Correct model transcribe+translate in {processing_time:.2f}s "
                f"({len(source_segments)} src, {len(target_segments)} tgt)"
            )

            return source_segments, target_segments

    # ------------------------------------------------------------------
    # Legacy single-model transcription (backward compat)
    # ------------------------------------------------------------------

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
        vad_model = self._load_vad_model()
        audio_tensor = torch.from_numpy(audio)
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            vad_model,
            sampling_rate=CONFIG.SAMPLE_RATE,
            threshold=CONFIG.VAD_THRESHOLD,
            min_speech_duration_ms=int(CONFIG.VAD_MIN_SPEECH_DURATION * 1000),
            min_silence_duration_ms=int(CONFIG.VAD_MIN_SILENCE_DURATION * 1000),
        )

        logger.info(f"VAD found {len(speech_timestamps)} speech timestamps")

        if len(speech_timestamps) == 0:
            # Fallback: if VAD finds nothing, try transcribing the whole audio
            # This handles continuous speech without clear silence boundaries
            logger.debug("VAD found 0 segments, trying direct transcription on full audio")
            if len(audio) >= CONFIG.SAMPLE_RATE * 0.5:  # At least 0.5s
                transcribe_kwargs = {
                    "language": self.language if self.language != "auto" else None,
                    "task": self.task,
                    "beam_size": beam_size,
                    "vad_filter": False,
                    "word_timestamps": False,
                    "no_speech_threshold": CONFIG.WHISPER_NO_SPEECH_THRESHOLD,
                    "log_prob_threshold": CONFIG.WHISPER_LOGPROB_THRESHOLD,
                    "compression_ratio_threshold": CONFIG.WHISPER_COMPRESSION_RATIO_THRESHOLD,
                }
                if use_context and initial_prompt:
                    transcribe_kwargs["initial_prompt"] = initial_prompt

                try:
                    segments, info = self._model.transcribe(audio, **transcribe_kwargs)
                    results = []
                    for segment in segments:
                        seg_start = segment.start
                        seg_end = segment.end
                        if window_start_time is not None:
                            seg_start += window_start_time
                            seg_end += window_start_time
                        results.append(
                            TranscriptionSegment(
                                start=seg_start,
                                end=seg_end,
                                text=segment.text.strip(),
                                language=info.language if info.language else self.language,
                                avg_logprob=(
                                    segment.avg_logprob
                                    if hasattr(segment, "avg_logprob")
                                    else -1.0
                                ),
                                no_speech_prob=(
                                    segment.no_speech_prob
                                    if hasattr(segment, "no_speech_prob")
                                    else 0.0
                                ),
                                audio_time_start=(
                                    seg_start if window_start_time is not None else None
                                ),
                                audio_time_end=(
                                    seg_end if window_start_time is not None else None
                                ),
                            )
                        )
                    if results:
                        logger.info(f"Direct transcription returned {len(results)} segments")
                    return results
                except Exception as e:
                    logger.error(f"Direct transcription failed: {e}")
            return []

        all_segments = []
        total_processing_time = 0.0

        for item in speech_timestamps:
            # Silero VAD returns timestamps in SAMPLES at the given sampling_rate
            start_sample = item["start"]
            end_sample = item["end"]
            start_time = start_sample / CONFIG.SAMPLE_RATE
            end_time = end_sample / CONFIG.SAMPLE_RATE

            audio_chunk = audio[start_sample:end_sample]

            if len(audio_chunk) < CONFIG.SAMPLE_RATE * 0.1:  # skip very short chunks
                continue

            # Transcribe this chunk without VAD filter (already filtered)
            start_transcribe = time.time()
            transcribe_kwargs = {
                "language": self.language if self.language != "auto" else None,
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
            segments_list = list(segments)
            logger.debug(f"Transcribed chunk {len(all_segments)+1}: got {len(segments_list)} segments")

            processing_time = time.time() - start_transcribe
            total_processing_time += processing_time

            for segment in segments_list:
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
            "fast_model_loaded": self._fast_model_loaded,
            "correct_model_loaded": self._correct_model_loaded,
        }

    def reset_stats(self) -> None:
        self._last_transcription_time = 0.0
        self._last_finish_time = 0.0
        self._total_processing_time = 0.0
        self._num_transcriptions = 0

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def is_fast_model_loaded(self) -> bool:
        return self._fast_model_loaded

    @property
    def is_correct_model_loaded(self) -> bool:
        return self._correct_model_loaded

    @property
    def are_models_loaded(self) -> bool:
        return self._fast_model_loaded and self._correct_model_loaded

    def is_busy(self) -> bool:
        with self._processing_lock:
            if self._is_processing:
                return True
            time_since_last = time.time() - self._last_finish_time
            return time_since_last < CONFIG.MIN_PROCESSING_INTERVAL

    def is_fast_busy(self) -> bool:
        """Check if the fast model is currently processing."""
        return not self._fast_processing_lock.acquire(blocking=False)
        # If we can't acquire, it's busy. Release if we did acquire.
        # Note: this is a best-effort check.

    def is_correct_busy(self) -> bool:
        """Check if the correct model is currently processing."""
        return not self._correct_processing_lock.acquire(blocking=False)

    def transcribe_and_translate(
        self,
        audio: np.ndarray,
        beam_size: int = CONFIG.WHISPER_BEAM_SIZE,
        window_start_time: Optional[float] = None,
    ) -> tuple[list[TranscriptionSegment], list[TranscriptionSegment]]:
        """Transcribe audio and translate to target language.

        Returns:
            Tuple of (source_segments, target_segments)
        """
        if not self._is_loaded:
            self.load_model()

        with self._processing_lock:
            self._is_processing = True
            try:
                return self._transcribe_and_translate_internal(audio, beam_size, window_start_time)
            finally:
                self._is_processing = False

    def _transcribe_and_translate_internal(
        self,
        audio: np.ndarray,
        beam_size: int = CONFIG.WHISPER_BEAM_SIZE,
        window_start_time: Optional[float] = None,
    ) -> tuple[list[TranscriptionSegment], list[TranscriptionSegment]]:
        """Internal method to transcribe and translate audio."""
        initial_prompt = self._build_initial_prompt()
        use_context = CONFIG.ENABLE_CONTEXT_CARRY and initial_prompt

        # Get speech timestamps using VAD
        vad_model = self._load_vad_model()
        audio_tensor = torch.from_numpy(audio)
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            vad_model,
            sampling_rate=CONFIG.SAMPLE_RATE,
            threshold=CONFIG.VAD_THRESHOLD,
            min_speech_duration_ms=int(CONFIG.VAD_MIN_SPEECH_DURATION * 1000),
            min_silence_duration_ms=int(CONFIG.VAD_MIN_SILENCE_DURATION * 1000),
        )

        logger.info(f"VAD found {len(speech_timestamps)} speech timestamps")

        if len(speech_timestamps) == 0:
            # Fallback: try direct transcription on full audio
            if len(audio) >= CONFIG.SAMPLE_RATE * 0.5:
                return self._transcribe_and_translate_chunk(
                    audio, beam_size, window_start_time or 0.0, use_context, initial_prompt
                )
            return [], []

        source_segments = []
        target_segments = []
        total_processing_time = 0.0

        for item in speech_timestamps:
            start_sample = item["start"]
            end_sample = item["end"]
            start_time = start_sample / CONFIG.SAMPLE_RATE
            end_time = end_sample / CONFIG.SAMPLE_RATE

            audio_chunk = audio[start_sample:end_sample]

            if len(audio_chunk) < CONFIG.SAMPLE_RATE * 0.1:
                continue

            chunk_window_time = (window_start_time or 0.0) + start_time
            src_segs, tgt_segs = self._transcribe_and_translate_chunk(
                audio_chunk, beam_size, chunk_window_time, use_context, initial_prompt
            )

            source_segments.extend(src_segs)
            target_segments.extend(tgt_segs)

        return source_segments, target_segments

    def _transcribe_and_translate_chunk(
        self,
        audio_chunk: np.ndarray,
        beam_size: int,
        window_start_time: float,
        use_context: bool,
        initial_prompt: Optional[str],
    ) -> tuple[list[TranscriptionSegment], list[TranscriptionSegment]]:
        """Transcribe a single chunk and translate it."""
        source_segments = []
        target_segments = []

        # Determine source language
        source_lang = self.language if self.language != "auto" else None

        # Step 1: Transcribe in source language
        transcribe_kwargs = {
            "language": source_lang,
            "task": "transcribe",
            "beam_size": beam_size,
            "vad_filter": False,
            "word_timestamps": False,
            "no_speech_threshold": CONFIG.WHISPER_NO_SPEECH_THRESHOLD,
            "log_prob_threshold": CONFIG.WHISPER_LOGPROB_THRESHOLD,
            "compression_ratio_threshold": CONFIG.WHISPER_COMPRESSION_RATIO_THRESHOLD,
        }
        if use_context and initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt

        try:
            segments, info = self._model.transcribe(audio_chunk, **transcribe_kwargs)
            segments_list = list(segments)
            detected_lang = info.language if info.language else self.language

            for segment in segments_list:
                seg_start = segment.start + window_start_time
                seg_end = segment.end + window_start_time
                source_segments.append(
                    TranscriptionSegment(
                        start=seg_start,
                        end=seg_end,
                        text=segment.text.strip(),
                        language=detected_lang,
                        avg_logprob=segment.avg_logprob if hasattr(segment, "avg_logprob") else -1.0,
                        no_speech_prob=segment.no_speech_prob if hasattr(segment, "no_speech_prob") else 0.0,
                    )
                )
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return [], []

        # Step 2: Translate to target language using the workaround
        if self.target_language and self.target_language != detected_lang:
            translate_kwargs = {
                "language": self.target_language,  # Force target language output
                "task": "transcribe",
                "beam_size": beam_size,
                "vad_filter": False,
                "word_timestamps": False,
                "no_speech_threshold": CONFIG.WHISPER_NO_SPEECH_THRESHOLD,
                "log_prob_threshold": CONFIG.WHISPER_LOGPROB_THRESHOLD,
                "compression_ratio_threshold": CONFIG.WHISPER_COMPRESSION_RATIO_THRESHOLD,
                "initial_prompt": f"Translate the following {detected_lang} speech to {self.target_language}.",
            }

            try:
                trans_segments, _ = self._model.transcribe(audio_chunk, **translate_kwargs)
                trans_list = list(trans_segments)

                for segment in trans_list:
                    seg_start = segment.start + window_start_time
                    seg_end = segment.end + window_start_time
                    target_segments.append(
                        TranscriptionSegment(
                            start=seg_start,
                            end=seg_end,
                            text=segment.text.strip(),
                            language=self.target_language,
                            avg_logprob=segment.avg_logprob if hasattr(segment, "avg_logprob") else -1.0,
                            no_speech_prob=segment.no_speech_prob if hasattr(segment, "no_speech_prob") else 0.0,
                        )
                    )
            except Exception as e:
                logger.error(f"Translation failed: {e}")
                # Use source segments as fallback
                for seg in source_segments:
                    target_segments.append(
                        TranscriptionSegment(
                            start=seg.start,
                            end=seg.end,
                            text=seg.text,
                            language=self.target_language,
                            avg_logprob=seg.avg_logprob,
                            no_speech_prob=seg.no_speech_prob,
                        )
                    )

        return source_segments, target_segments
