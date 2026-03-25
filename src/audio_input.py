import logging
import threading
from typing import Any, Callable, Optional

import numpy as np
import sounddevice as sd
from faster_whisper.vad import VadOptions

from .config import CONFIG

logger = logging.getLogger(__name__)


class AudioInput:
    def __init__(
        self,
        sample_rate: int = CONFIG.SAMPLE_RATE,
        channels: int = CONFIG.CHANNELS,
        dtype: str = CONFIG.DTYPE,
        buffer_duration: float = CONFIG.BUFFER_DURATION,
        chunk_duration: float = CONFIG.CHUNK_DURATION,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self._device = device
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.buffer_samples = int(sample_rate * buffer_duration)

        self._buffer: np.ndarray = np.zeros(self.buffer_samples, dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._buffer_pos = 0
        self._buffer_filled = 0
        self._total_samples_processed = 0

        self._stream: Optional[sd.InputStream] = None
        self._is_running = False
        self._callback: Optional[Callable[[np.ndarray], Any]] = None

        self._vad_options = VadOptions(
            threshold=CONFIG.VAD_THRESHOLD,
            min_speech_duration_ms=int(CONFIG.VAD_MIN_SPEECH_DURATION * 1000),
            min_silence_duration_ms=int(CONFIG.VAD_MIN_SILENCE_DURATION * 1000),
        )

    @property
    def vad_options(self) -> VadOptions:
        return self._vad_options

    def set_callback(self, callback: Callable[[np.ndarray], Any]) -> None:
        self._callback = callback

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags
    ) -> None:
        if status:
            logger.warning(f"Audio callback status: {status}")

        audio_chunk = indata[:, 0].astype(np.float32)

        with self._buffer_lock:
            self._total_samples_processed += frames
            if self._buffer_pos + frames <= self.buffer_samples:
                self._buffer[self._buffer_pos : self._buffer_pos + frames] = audio_chunk
                self._buffer_pos += frames
            else:
                remaining = self.buffer_samples - self._buffer_pos
                self._buffer[self._buffer_pos :] = audio_chunk[:remaining]
                self._buffer[: frames - remaining] = audio_chunk[remaining:]
                self._buffer_pos = frames - remaining

            if self._buffer_filled < self.buffer_samples:
                self._buffer_filled = min(self._buffer_filled + frames, self.buffer_samples)

        if self._callback:
            try:
                self._callback(audio_chunk)
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")

    def get_buffer(self) -> np.ndarray:
        with self._buffer_lock:
            if self._buffer_filled < self.buffer_samples:
                return self._buffer[: self._buffer_filled].copy()
            return self._buffer.copy()

    def get_audio_window(self, start_seconds: float, duration: float) -> np.ndarray:
        start_samples = int(start_seconds * self.sample_rate)
        num_samples = int(duration * self.sample_rate)

        with self._buffer_lock:
            if start_samples >= self._buffer_filled:
                return np.zeros(num_samples, dtype=np.float32)

            if start_samples + num_samples <= self.buffer_samples:
                end_pos = min(start_samples + num_samples, self._buffer_filled)
                result = self._buffer[start_samples:end_pos]
                if len(result) < num_samples:
                    result = np.pad(result, (0, num_samples - len(result)))
                return result.copy()

            part1_len = self.buffer_samples - start_samples
            part2_len = num_samples - part1_len
            result = np.concatenate([self._buffer[start_samples:], self._buffer[:part2_len]])
            return result

    def get_recent_audio(self, duration: float) -> np.ndarray:
        num_samples = int(duration * self.sample_rate)

        with self._buffer_lock:
            if self._buffer_filled < num_samples:
                return self._buffer[: self._buffer_filled].copy()

            if self._buffer_pos >= num_samples:
                return self._buffer[self._buffer_pos - num_samples : self._buffer_pos].copy()
            else:
                part1_len = self.buffer_samples - (num_samples - self._buffer_pos)
                return np.concatenate(
                    [self._buffer[part1_len:], self._buffer[: self._buffer_pos]]
                ).copy()

    def start(self) -> None:
        if self._is_running:
            logger.warning("AudioInput already running")
            return

        self._buffer = np.zeros(self.buffer_samples, dtype=np.float32)
        self._buffer_pos = 0
        self._buffer_filled = 0

        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.chunk_samples,
            callback=self._audio_callback,
        )

        self._stream.start()
        self._is_running = True
        logger.info("AudioInput started")

    def stop(self) -> None:
        if not self._is_running:
            return

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._is_running = False
        logger.info("AudioInput stopped")

    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_current_timestamp(self) -> float:
        with self._buffer_lock:
            return self._total_samples_processed / self.sample_rate
