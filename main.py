#!/usr/bin/env python3
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Patch httpcore to use unverified SSL for TLS connections
import httpcore._backends.sync as sync_backend

# Check if TLSinTLSStream exists and patch it
if hasattr(sync_backend, 'TLSinTLSStream'):
    _original_tls_start = sync_backend.TLSinTLSStream.start_tls

    def _patched_tls_start(self, ssl_context, timeout=None):
        if ssl_context is None:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        return _original_tls_start(self, ssl_context, timeout)

    sync_backend.TLSinTLSStream.start_tls = _patched_tls_start

"""
Real-Time Speech Translation System
Korean -> English translation using Faster-Whisper with speaker diarization
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import threading
import time
import base64
import struct
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import websockets
from src.audio_input import AudioInput
from src.config import CONFIG
from src.diarization import DiarizationHandler, SpeakerSegment
from src.fusion import Fusion
from gradio_gui import GradioGUI
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
        audio_device: Optional[int] = None,
    ):
        self.enable_gui = enable_gui
        self.enable_diarization = enable_diarization
        self.gui_port = gui_port
        self.benchmark_mode = benchmark_mode
        self.audio_device = audio_device

        self._running = False
        self._shutdown_event = threading.Event()
        self._paused = False

        self._audio_input: Optional[AudioInput] = None
        self._whisper: Optional[WhisperASR] = None
        self._diarization: Optional[DiarizationHandler] = None
        self._fusion: Optional[Fusion] = None
        self._gui: Optional[GradioGUI] = None

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

        self._audio_input = AudioInput(device=self.audio_device)
        logger.info("AudioInput initialized")

        self._whisper = WhisperASR()
        logger.info("WhisperASR initialized")

        if self.enable_diarization:
            self._diarization = DiarizationHandler()
            logger.info("DiarizationHandler initialized")

        self._fusion = Fusion()
        logger.info("Fusion initialized")

        if self.enable_gui:
            self._gui = GradioGUI(self._fusion, pipeline=self)
            logger.info("GradioGUI initialized")

        self._ws_port = 8765
        self._audio_buffer = []
        self._source_language = "auto"
        self._target_language = "en"
        self._ws_running = False
        self._cached_speaker_segments: list[SpeakerSegment] = []

    def set_languages(self, source_lang: str, target_lang: str):
        """Set source and target languages for transcription."""
        self._source_language = source_lang
        self._target_language = target_lang
        logger.info(f"Languages set: {source_lang} -> {target_lang}")

    async def _handle_websocket(self, websocket, path):
        client_ip = websocket.remote_address
        logger.info(f"WebSocket client connected: {client_ip}")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "start":
                        self._source_language = data.get("sourceLang", "auto")
                        self._target_language = data.get("targetLang", "en")
                        logger.info(f"Starting WS transcription: {self._source_language} -> {self._target_language}")

                        if hasattr(self, "set_languages"):
                            self.set_languages(self._source_language, self._target_language)

                        if self._whisper and not self._whisper.is_loaded():
                            self._whisper.load_model()
                            if self._diarization:
                                self._diarization.load_pipeline()

                        self._audio_buffer = []
                        await websocket.send(json.dumps({"type": "started"}))

                    elif msg_type == "audio_chunk":
                        audio_data = data.get("data")
                        if audio_data:
                            try:
                                audio_bytes = base64.b64decode(audio_data)
                                audio_array = struct.unpack(f"{len(audio_bytes)//4}f", audio_bytes)
                                self._audio_buffer.extend(audio_array)

                                if len(self._audio_buffer) >= CONFIG.SAMPLE_RATE * 3:
                                    await self._process_ws_audio(websocket)
                            except Exception as e:
                                logger.error(f"Audio decode error: {e}")

                    elif msg_type == "stop":
                        logger.info("Stopping WS transcription")
                        if self._audio_buffer:
                            await self._process_ws_audio(websocket)
                        self._audio_buffer = []
                        await websocket.send(json.dumps({"type": "stopped"}))

                except json.JSONDecodeError:
                    logger.error("Invalid JSON message")
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    await websocket.send(json.dumps({"type": "error", "message": str(e)}))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket client disconnected: {client_ip}")
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")

    async def _process_ws_audio(self, websocket):
        if not self._audio_buffer or len(self._audio_buffer) < CONFIG.SAMPLE_RATE * 0.5:
            return

        audio = np.array(self._audio_buffer, dtype=np.float32)
        self._audio_buffer = []

        max_amp = np.max(np.abs(audio))
        if max_amp < 0.005:
            return

        window_start_time = time.time() - 3.0

        timestamp_start = time.time()
        asr_segments, translated_segments = self._whisper.transcribe_and_translate(
            audio,
            window_start_time=window_start_time,
        )
        processing_time = time.time() - timestamp_start

        if not asr_segments:
            return

        fused_segments = self._fusion.fuse(
            asr_segments,
            translated_segments,
            self._cached_speaker_segments if self._diarization else [],
            timestamp_start,
            self._source_language,
            self._target_language,
        )

        latency = processing_time
        rtf = processing_time / 3.0 if audio.shape[0] > 0 else 0

        segments_data = []
        for seg in fused_segments:
            segments_data.append({
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker,
                "source_text": seg.source_text,
                "target_text": seg.target_text,
                "source_language": seg.source_language,
                "target_language": seg.target_language,
            })

        stats = self._fusion.get_stats()
        metrics = {
            "latency": latency,
            "rtf": rtf,
            "segments": stats["total_segments"],
            "speakers": stats["unique_speakers"],
        }

        await websocket.send(json.dumps({
            "type": "transcript",
            "segments": segments_data,
            "metrics": metrics,
        }))

    async def _run_websocket_server(self):
        logger.info(f"Starting WebSocket server on port {self._ws_port}")
        self._ws_running = True
        async with websockets.serve(self._handle_websocket, "0.0.0.0", self._ws_port):
            while self._ws_running:
                await asyncio.sleep(1)
        logger.info("WebSocket server stopped")

    def start_websocket_server(self):
        if not hasattr(self, '_ws_thread') or not self._ws_thread.is_alive():
            self._ws_thread = threading.Thread(target=self._run_ws_event_loop, daemon=True)
            self._ws_thread.start()
            logger.info("WebSocket server thread started")

    def _run_ws_event_loop(self):
        asyncio.run(self._run_websocket_server())

    def start(self) -> None:
        if self._running:
            logger.warning("Pipeline already running")
            return

        logger.info("Starting translation pipeline...")

        self._whisper.load_model()

        if self._diarization:
            self._diarization.load_pipeline()

        self._audio_input.start()

        # GUI is already served by launcher, no need to serve again
        # if self._gui:
        #     self._gui.serve(port=self.gui_port)

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

        self._process_interval = 2.0
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
            logger.debug("Audio input not running")
            return

        if self._whisper.is_busy():
            logger.debug("Whisper is busy, skipping this cycle")
            return

        audio_duration = 3.0
        window_start_time = self._audio_input.get_current_timestamp() - audio_duration

        if window_start_time < 0:
            window_start_time = 0.0

        audio = self._audio_input.get_recent_audio(audio_duration)

        if len(audio) < CONFIG.SAMPLE_RATE * 0.5:
            logger.debug(f"Audio too short: {len(audio)} samples")
            return

        max_amp = max(abs(x) for x in audio) if len(audio) > 0 else 0
        logger.info(f"Audio buffer: {len(audio)} samples, max_amp={max_amp:.4f}")

        # Skip if audio is too quiet (below threshold)
        if max_amp < 0.005:  # Lower threshold for loopback audio
            logger.debug(f"Audio too quiet (max_amp={max_amp:.4f}), skipping")
            return

        # Start metric tracking
        timestamp_start = time.time()

        logger.info("Starting transcription and translation...")
        asr_segments, translated_segments = self._whisper.transcribe_and_translate(
            audio,
            window_start_time=window_start_time,
        )

        logger.info(f"Transcription returned {len(asr_segments)} source, {len(translated_segments)} target segments")

        if not asr_segments:
            logger.debug("No ASR segments found, returning")
            return

        # Filter out segments that have already been emitted
        # Use a small epsilon to avoid floating point issues
        epsilon = 0.1
        new_asr_segments = [
            s for s in asr_segments if s.end > self._last_emitted_end_time + epsilon
        ]
        new_translated_segments = [
            s for s in translated_segments if s.end > self._last_emitted_end_time + epsilon
        ]

        logger.info(f"New segments to emit: {len(new_asr_segments)} source, {len(new_translated_segments)} target")

        if not new_asr_segments:
            logger.debug("No new segments to emit (all already emitted)")
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

        source_lang = self._whisper.language if self._whisper.language != "auto" else "auto"
        target_lang = self._whisper.target_language or "en"

        logger.info(f"Fusing {len(new_asr_segments)} ASR segments with {len(speaker_segments)} speaker segments")
        fused_segments = self._fusion.fuse(
            new_asr_segments,
            new_translated_segments,
            speaker_segments,
            timestamp_start,
            source_language=source_lang,
            target_language=target_lang,
        )

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

        # Log metrics
        log_msg = (
            f"METRICS: latency={latency_seconds:.2f}s "
            f"processing={last_processing_time:.2f}s "
            f"audio_duration={audio_duration:.2f}s "
            f"rtf={rtf:.2f} segments={segments_emitted}"
        )
        logger.info(log_msg)

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

    def set_audio_device(self, device: int) -> None:
        """Set the audio input device"""
        if self._running:
            logger.warning("Cannot change device while running")
            return
        self.audio_device = device
        if self._audio_input:
            self._audio_input._device = device

    def set_languages(self, source_language: str, target_language: str) -> None:
        """Set source and target languages for transcription and translation."""
        if self._whisper:
            self._whisper.set_languages(source_language, target_language)
        logger.info(f"Languages set: source={source_language}, target={target_language}")

    @property
    def is_running(self) -> bool:
        return self._running

    def serve(self, **kwargs) -> None:
        if self._gui:
            self._gui.serve(port=self.gui_port, **kwargs)


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
        # Don't auto-start - GUI will control when to start
        logger.info("GUI is ready. Click 'Start Translation' to begin.")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        if pipeline.is_running:
            pipeline.stop()


if __name__ == "__main__":
    main()
