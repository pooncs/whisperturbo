#!/usr/bin/env python3
"""
Real-Time Speech Translation System
Korean -> English translation using Faster-Whisper with speaker diarization
"""

import os
import sys
import time
import signal
import logging
import argparse
import threading
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG
from src.audio_input import AudioInput
from src.whisper_asr import WhisperASR
from src.diarization import DiarizationHandler
from src.fusion import Fusion
from src.gui import TranslationGUI


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
    ):
        self.enable_gui = enable_gui
        self.enable_diarization = enable_diarization
        self.gui_port = gui_port

        self._running = False
        self._shutdown_event = threading.Event()

        self._audio_input: Optional[AudioInput] = None
        self._whisper: Optional[WhisperASR] = None
        self._diarization: Optional[DiarizationHandler] = None
        self._fusion: Optional[Fusion] = None
        self._gui: Optional[TranslationGUI] = None

        self._process_thread: Optional[threading.Thread] = None

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
            self._gui = TranslationGUI(self._fusion)
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

        process_interval = 3.0
        last_process_time = time.time()

        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()

                if current_time - last_process_time >= process_interval:
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

        audio_duration = 5.0
        audio = self._audio_input.get_recent_audio(audio_duration)

        if len(audio) < CONFIG.SAMPLE_RATE * 0.5:
            return

        timestamp = time.time()

        asr_segments = self._whisper.transcribe(
            audio,
            vad_options=self._audio_input.vad_options,
        )

        if not asr_segments:
            return

        speaker_segments = []

        if self._diarization and self._diarization.is_loaded:
            try:
                audio_for_diarization = self._audio_input.get_recent_audio(
                    CONFIG.DIARIZATION_WINDOW_SIZE
                )
                diarization_timestamp = (
                    self._audio_input.get_current_timestamp()
                    - CONFIG.DIARIZATION_WINDOW_SIZE
                )
                speaker_segments = self._diarization.diarize_audio(
                    audio_for_diarization,
                    diarization_timestamp,
                )
            except Exception as e:
                logger.error(f"Diarization error: {e}")

        fused_segments = self._fusion.fuse(asr_segments, speaker_segments, timestamp)

        if self._gui and fused_segments:
            self._gui.add_segments(fused_segments)

            stats = self._whisper.get_stats()
            latency = time.time() - timestamp
            rtf = (
                stats.get("last_processing_time", 0) / audio_duration
                if audio_duration > 0
                else 0
            )

            processing_rate = (
                len(fused_segments) / process_interval if process_interval > 0 else 0
            )
            self._gui.update_kpis(latency, rtf, processing_rate)

        logger.info(
            f"Processed {len(asr_segments)} ASR segments, {len(speaker_segments)} speaker segments, {len(fused_segments)} fused segments"
        )

    @property
    def is_running(self) -> bool:
        return self._running


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Korean -> English Speech Translation"
    )
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
