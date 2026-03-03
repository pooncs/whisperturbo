#!/usr/bin/env python3
"""
Real-Time Speech Translation System
Korean -> English translation using Faster-Whisper with speaker diarization
"""

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_input import AudioInput
from src.config import CONFIG
from src.diarization import DiarizationHandler, SpeakerSegment
from src.fusion import Fusion
from src.gui import TranslationGUI
from src.whisper_asr import WhisperASR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TranslationPipeline:
    def __init__(
        self,
        enable_gui: bool = True,
        enable_diarization: bool = True,
        gui_port: int = 5006,
        benchmark_mode: bool = False,
    ):
        self.enable_gui = enable_gui
        self.enable_diarization = enable_diarization
        self.gui_port = gui_port
        self.benchmark_mode = benchmark_mode

        self._running = False
        self._shutdown_event = threading.Event()
        self._paused = False

        self._audio_input: Optional[AudioInput] = None
        self._whisper: Optional[WhisperASR] = None
        self._diarization: Optional[DiarizationHandler] = None
        self._fusion: Optional[Fusion] = None
        self._gui: Optional[TranslationGUI] = None

        self._process_thread: Optional[threading.Thread] = None

        self._audio_time_offset = 0.0
        self._process_interval = 3.0

        self._diarization_interval = 15.0
        self._last_diarization_time = 0.0
        self._cached_speaker_segments: list[SpeakerSegment] = []

        self._last_emitted_end_time = 0.0

        # Metrics tracking
        self._metrics = {
            "latencies": [],
            "rtfs": [],
            "processing_times": [],
            "total_segments": 0,
            "cycles": 0,
        }

        self._setup_components()

    def _setup_components(self) -> None:
        logger.info("Setting up translation pipeline...")

        self._audio_input = AudioInput()
        logger.info("AudioInput initialized")

        self._whisper = WhisperASR()
        logger.info("WhisperASR initialized")

        if self.enable_diarization:
            self._diarization = DiarizationHandler()
            logger.info("DiarizationHandler initialized")

        self._fusion = Fusion()
        logger.info("Fusion initialized")

        if self.enable_gui:
            self._gui = TranslationGUI(self._fusion, pipeline=self)
            logger.info("TranslationGUI initialized")

    def start(self) -> None:
        if self._running:
            logger.warning("Pipeline already running")
            return

        logger.info("Starting translation pipeline...")

        self._whisper.load_model()

        if self._diarization:
            self._diarization.load_pipeline()

        self._audio_input.start()

        if self._gui:
            self._gui.serve(port=self.gui_port)

        self._running = True
        self._shutdown_event.clear()

        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

        logger.info("Translation pipeline started successfully")

    def stop(self) -> None:
        if not self._running:
            return

        logger.info("Stopping translation pipeline...")

        self._shutdown_event.set()

        if self._process_thread:
            self._process_thread.join(timeout=5.0)

        self._audio_input.stop()

        if self._diarization:
            self._diarization.unload_pipeline()

        self._whisper.unload_model()

        if self._gui:
            self._gui.stop()

        self._running = False

        logger.info("Translation pipeline stopped")

    def _process_loop(self) -> None:
        logger.info("Starting processing loop...")

        self._process_interval = 3.0
        last_process_time = time.time()

        while not self._shutdown_event.is_set():
            try:
                if self._paused:
                    time.sleep(0.1)
                    continue

                current_time = time.time()

                if current_time - last_process_time >= self._process_interval:
                    self._process_audio()
                    last_process_time = current_time

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(1.0)

        logger.info("Processing loop ended")

    def _process_audio(self) -> None:
        if not self._audio_input.is_running:
            return

        if self._whisper.is_busy():
            logger.debug("Whisper is busy, skipping this cycle")
            return

        audio_duration = 5.0
        window_start_time = self._audio_input.get_current_timestamp() - audio_duration

        if window_start_time < 0:
            window_start_time = 0.0

        audio = self._audio_input.get_recent_audio(audio_duration)

        if len(audio) < CONFIG.SAMPLE_RATE * 0.5:
            return

        # Start metric tracking
        timestamp_start = time.time()

        asr_segments = self._whisper.transcribe_vad_chunks(
            audio,
            window_start_time=window_start_time,
        )

        if not asr_segments:
            return

        # Filter out segments that have already been emitted
        # Use a small epsilon to avoid floating point issues
        epsilon = 0.1
        new_asr_segments = [
            s for s in asr_segments if s.end > self._last_emitted_end_time + epsilon
        ]

        if not new_asr_segments:
            return

        # Handle async diarization
        current_time = time.time()
        if (
            self._diarization
            and self._diarization.is_loaded
            and current_time - self._last_diarization_time >= self._diarization_interval
        ):
            if self._diarization.is_busy():
                logger.warning("Diarization is busy, skipping this cycle")
            else:
                audio_for_diarization = self._audio_input.get_recent_audio(
                    CONFIG.DIARIZATION_WINDOW_SIZE
                )
                diarization_timestamp = (
                    self._audio_input.get_current_timestamp() - CONFIG.DIARIZATION_WINDOW_SIZE
                )
                self._diarization.process_async(audio_for_diarization, diarization_timestamp)
                self._last_diarization_time = current_time

        # Get latest diarization results from cache
        if self._diarization:
            latest_results = self._diarization.get_latest_results()
            if latest_results is not None:
                self._cached_speaker_segments = latest_results

        speaker_segments = self._cached_speaker_segments

        fused_segments = self._fusion.fuse(new_asr_segments, speaker_segments, timestamp_start)

        if fused_segments:
            self._last_emitted_end_time = max(s.end for s in fused_segments)

        stats = self._whisper.get_stats()
        last_processing_time = stats.get("last_processing_time", 0)
        latency_seconds = time.time() - timestamp_start
        rtf = last_processing_time / audio_duration if audio_duration > 0 else 0
        segments_emitted = len(fused_segments)

        # Update aggregate metrics
        self._metrics["latencies"].append(latency_seconds)
        self._metrics["rtfs"].append(rtf)
        self._metrics["processing_times"].append(last_processing_time)
        self._metrics["total_segments"] += segments_emitted
        self._metrics["cycles"] += 1

        # Log metrics consistently
        log_msg = (
            f"METRICS: latency={latency_seconds:.2f}s "
            f"processing={last_processing_time:.2f}s "
            f"audio_duration={audio_duration:.2f}s "
            f"rtf={rtf:.2f} segments={segments_emitted}"
        )

        if logger.isEnabledFor(logging.DEBUG) or self.benchmark_mode:
            logger.info(log_msg)
        else:
            logger.debug(log_msg)

        if self.benchmark_mode and self._metrics["cycles"] % 5 == 0:
            avg_latency = sum(self._metrics["latencies"]) / len(self._metrics["latencies"])
            avg_rtf = sum(self._metrics["rtfs"]) / len(self._metrics["rtfs"])
            logger.info(
                f"BENCHMARK: Avg Latency: {avg_latency:.2f}s, Avg RTF: {avg_rtf:.2f}x, "
                f"Total Segments: {self._metrics['total_segments']}"
            )

        if self._gui and fused_segments:
            processing_rate = (
                len(fused_segments) / self._process_interval if self._process_interval > 0 else 0
            )
            self._gui.update_kpis(latency_seconds, rtf, processing_rate)
            self._gui.add_segments(fused_segments)

        logger.info(
            f"Processed {len(asr_segments)} ASR segments, {len(speaker_segments)} speaker segments, {len(fused_segments)} fused segments"
        )

    def pause(self) -> None:
        self._paused = True
        logger.info("Pipeline paused")

    def resume(self) -> None:
        self._paused = False
        logger.info("Pipeline resumed")

    @property
    def is_running(self) -> bool:
        return self._running


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Korean -> English Speech Translation")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable GUI",
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization",
    )
    parser.add_argument(
        "--gui-port",
        type=int,
        default=5006,
        help="GUI server port (default: 5006)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run in benchmark mode (more verbose metrics)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if not CONFIG.hf_token and args.no_diarization:
        logger.warning(
            "HF_TOKEN not set. Speaker diarization requires authentication. "
            "Use --no-diarization to disable."
        )

    pipeline = TranslationPipeline(
        enable_gui=not args.no_gui,
        enable_diarization=not args.no_diarization,
        gui_port=args.gui_port,
        benchmark_mode=args.benchmark,
    )

    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        pipeline.start()

        logger.info("Pipeline running. Press Ctrl+C to stop.")

        while pipeline.is_running:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
