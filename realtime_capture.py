"""
Real-Time Audio Streaming Capture with Continuous Translation
Captures audio for 10 minutes, transcribes, translates, diarizes,
and writes continuously to a markdown file with timestamps.
"""

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal
from faster_whisper import WhisperModel

# Set HF_TOKEN for pyannote
DEFAULT_HF_TOKEN = ""
if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = DEFAULT_HF_TOKEN

from pyannote.audio import Pipeline
import torch

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = r"C:\Users\hmgics\projects\test_outputs"
DEFAULT_DURATION = 600  # 10 minutes


class RealTimeCapture:
    def __init__(
        self,
        device: int = 1,
        device_name: str = "CABLE Output (VB-Audio Virtual Cable)",
        duration: float = DEFAULT_DURATION,
        sample_rate: int = 16000,
        chunk_duration: float = 3.0,
        overlap: float = 0.5,
    ):
        self.device = device
        self.device_name = device_name
        self.duration = duration
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.overlap_samples = int(sample_rate * overlap)

        # Models
        self.whisper: Optional[WhisperModel] = None
        self.diarization_pipeline: Optional[Pipeline] = None

        # Output
        self.output_path: Optional[str] = None
        self.transcript_path: Optional[str] = None
        self.file_handle = None
        self.file_lock = threading.Lock()

        # State
        self.start_time = 0.0
        self.total_audio_samples = 0
        self.segments_written = 0
        self.is_running = False
        self.shutdown_event = threading.Event()

        # Audio buffer
        self.audio_buffer: list[np.ndarray] = []
        self.buffer_lock = threading.Lock()

        # Stats
        self.stats = {
            "transcribe_time": 0.0,
            "translate_time": 0.0,
            "diarize_time": 0.0,
            "total_segments": 0,
            "total_audio_seconds": 0.0,
            "start_wall_time": 0.0,
        }

    def load_models(self):
        """Load Whisper and diarization models."""
        print("Loading Whisper model (large-v3-turbo, CTranslate2)...")
        t0 = time.time()
        self.whisper = WhisperModel(
            "large-v3-turbo", device="cuda", compute_type="int8_float16"
        )
        print(f"  Whisper loaded in {time.time() - t0:.2f}s")

        print("Loading diarization model...")
        t0 = time.time()
        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=True,
            )
            self.diarization_pipeline.to(torch.device("cuda"))
            print(f"  Diarization loaded in {time.time() - t0:.2f}s")
        except Exception as e:
            print(f"  Diarization failed to load: {e}")
            self.diarization_pipeline = None

    def setup_output_files(self):
        """Create output markdown and transcript files."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_path = os.path.join(OUTPUT_DIR, f"realtime_{timestamp}.md")
        self.transcript_path = os.path.join(
            OUTPUT_DIR, f"realtime_{timestamp}_transcript.md"
        )

        # Write header
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(f"# Real-Time Translation Session\n\n")
            f.write(f"**Device**: {self.device_name} (device {self.device})\n")
            f.write(f"**Started**: {datetime.now()}\n")
            f.write(f"**Duration**: {self.duration}s\n")
            f.write(f"**Sample Rate**: {self.sample_rate}Hz\n\n")
            f.write(f"---\n\n")

        with open(self.transcript_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcript - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"| Time | Speaker | Text |\n")
            f.write(f"|------|---------|------|\n")

        print(f"Output: {self.output_path}")
        print(f"Transcript: {self.transcript_path}")

    def write_segment(
        self, segment_text: str, speaker: str = "Unknown", timestamp: float = 0.0
    ):
        """Write a segment to the transcript file."""
        time_str = f"{timestamp:.1f}s"
        with self.file_lock:
            with open(self.transcript_path, "a", encoding="utf-8") as f:
                f.write(f"| {time_str} | {speaker} | {segment_text} |\n")
            self.segments_written += 1

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio input callback."""
        if status:
            print(f"Audio status: {status}")

        audio_chunk = indata[:, 0].astype(np.float32)
        with self.buffer_lock:
            self.audio_buffer.append(audio_chunk.copy())
            self.total_audio_samples += frames

    def get_recent_audio(self, duration: float) -> np.ndarray:
        """Get audio from the last N seconds."""
        target_samples = int(duration * self.sample_rate)
        with self.buffer_lock:
            if not self.audio_buffer:
                return np.zeros(target_samples, dtype=np.float32)

            # Concatenate buffer
            all_audio = np.concatenate(self.audio_buffer)

            if len(all_audio) >= target_samples:
                return all_audio[-target_samples:]
            else:
                return np.pad(all_audio, (0, target_samples - len(all_audio)))

    def transcribe_chunk(self, audio: np.ndarray, chunk_start: float) -> list[dict]:
        """Transcribe a chunk of audio."""
        if len(audio) < self.sample_rate * 0.3:
            return []

        # Skip very quiet audio
        max_amp = np.max(np.abs(audio))
        if max_amp < 0.005:
            return []

        # Transcribe in Korean
        t0 = time.time()
        segments_ko, info = self.whisper.transcribe(
            audio,
            language="ko",
            task="transcribe",
            beam_size=3,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        segments_ko = list(segments_ko)
        self.stats["transcribe_time"] += time.time() - t0

        if not segments_ko:
            return []

        # Translate to English using workaround
        t0 = time.time()
        try:
            segments_en, _ = self.whisper.transcribe(
                audio,
                language="en",
                task="transcribe",
                initial_prompt="Translate Korean speech to English.",
                beam_size=3,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            segments_en = list(segments_en)
        except Exception:
            segments_en = segments_ko
        self.stats["translate_time"] += time.time() - t0

        # Build results
        results = []
        for i, seg_ko in enumerate(segments_ko):
            seg_en = segments_en[i] if i < len(segments_en) else None
            start = seg_ko.start + chunk_start
            end = seg_ko.end + chunk_start

            results.append(
                {
                    "start": start,
                    "end": end,
                    "korean": seg_ko.text.strip(),
                    "english": seg_en.text.strip() if seg_en else "",
                }
            )

        return results

    def diarize_chunk(self, audio: np.ndarray, chunk_start: float) -> list[dict]:
        """Diarize a chunk of audio."""
        if not self.diarization_pipeline:
            return []

        if len(audio) < self.sample_rate * 2:
            return []

        t0 = time.time()
        try:
            audio_dict = {
                "waveform": torch.from_numpy(audio).float().unsqueeze(0),
                "sample_rate": self.sample_rate,
            }
            diarization = self.diarization_pipeline(audio_dict)
            annotation = diarization.speaker_diarization

            results = []
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                results.append(
                    {
                        "start": turn.start + chunk_start,
                        "end": turn.end + chunk_start,
                        "speaker": speaker,
                    }
                )
        except Exception as e:
            results = []
        self.stats["diarize_time"] += time.time() - t0

        return results

    def match_speaker(
        self, seg_start: float, seg_end: float, diarization: list[dict]
    ) -> str:
        """Match a transcription segment to a speaker."""
        if not diarization:
            return "Unknown"

        seg_mid = (seg_start + seg_end) / 2
        for spk in diarization:
            if spk["start"] <= seg_mid <= spk["end"]:
                return spk["speaker"]
        return "Unknown"

    def process_chunk(self):
        """Process the current audio chunk."""
        elapsed = self.total_audio_samples / self.sample_rate
        audio = self.get_recent_audio(self.chunk_duration)

        # Get diarization for a longer window
        diarization_window = self.get_recent_audio(min(15.0, elapsed))
        diarization_offset = max(0, elapsed - 15.0)

        # Run diarization on longer window
        diarization = self.diarize_chunk(diarization_window, diarization_offset)

        # Transcribe on shorter chunk
        chunk_offset = max(0, elapsed - self.chunk_duration)
        segments = self.transcribe_chunk(audio, chunk_offset)

        # Match speakers and write
        for seg in segments:
            speaker = self.match_speaker(seg["start"], seg["end"], diarization)
            self.write_segment(seg["english"], speaker, seg["start"])
            self.stats["total_segments"] += 1

        return len(segments)

    def print_stats(self, elapsed: float, wall_time: float):
        """Print throughput statistics."""
        transcribe_rtf = (
            elapsed / self.stats["transcribe_time"]
            if self.stats["transcribe_time"] > 0
            else 0
        )
        translate_rtf = (
            elapsed / self.stats["translate_time"]
            if self.stats["translate_time"] > 0
            else 0
        )
        diarize_rtf = (
            elapsed / self.stats["diarize_time"]
            if self.stats["diarize_time"] > 0
            else 0
        )

        print(f"\n{'='*60}")
        print(f"THROUGHPUT (at {elapsed:.0f}s / {wall_time:.1f}s wall)")
        print(f"{'='*60}")
        print(f"Segments:     {self.stats['total_segments']}")
        print(
            f"Transcribe:   {self.stats['transcribe_time']:.1f}s ({transcribe_rtf:.1f}x RT)"
        )
        print(
            f"Translate:    {self.stats['translate_time']:.1f}s ({translate_rtf:.1f}x RT)"
        )
        print(
            f"Diarize:      {self.stats['diarize_time']:.1f}s ({diarize_rtf:.1f}x RT)"
        )
        print(f"Wall time:    {wall_time:.1f}s")
        print(f"Real-time factor: {elapsed/wall_time:.1f}x")
        print(f"{'='*60}")

    def run(self):
        """Run the real-time capture."""
        print("=" * 60)
        print("Real-Time Audio Streaming Capture")
        print("=" * 60)
        print(f"Device: {self.device_name}")
        print(f"Duration: {self.duration}s ({self.duration/60:.0f} min)")
        print(f"Chunk: {self.chunk_duration}s with {self.overlap}s overlap")
        print()

        # Load models
        self.load_models()
        print()

        # Setup output
        self.setup_output_files()
        print()

        # Start audio capture
        print("Starting audio capture...")
        stream = sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=int(self.sample_rate * 0.5),  # 0.5s blocks
            callback=self._audio_callback,
        )
        stream.start()
        self.start_time = time.time()
        self.stats["start_wall_time"] = self.start_time
        self.is_running = True

        print(f"Recording for {self.duration}s. Press Ctrl+C to stop early.")
        print()

        try:
            last_process_time = 0.0
            last_stats_time = self.start_time

            while time.time() - self.start_time < self.duration:
                if self.shutdown_event.is_set():
                    break

                elapsed = time.time() - self.start_time

                # Process every chunk_duration seconds
                if elapsed - last_process_time >= self.chunk_duration:
                    segs = self.process_chunk()
                    last_process_time = elapsed

                    if segs > 0:
                        print(f"  [{elapsed:.0f}s] Processed {segs} segments")

                # Print stats every 30s
                wall_time = time.time() - self.start_time
                if wall_time - (last_stats_time - self.start_time) >= 30:
                    self.print_stats(elapsed, wall_time)
                    last_stats_time = time.time()

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.is_running = False
            stream.stop()
            stream.close()

            # Write final stats
            wall_time = time.time() - self.start_time
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(f"\n---\n\n")
                f.write(f"**Ended**: {datetime.now()}\n")
                f.write(f"**Total Duration**: {self.start_time:.1f}s\n")
                f.write(f"**Wall Time**: {wall_time:.1f}s\n")
                f.write(f"**Total Segments**: {self.stats['total_segments']}\n")
                f.write(
                    f"**Transcribe RTF**: {self.duration/self.stats['transcribe_time']:.1f}x\n"
                )
                f.write(
                    f"**Translate RTF**: {self.duration/self.stats['translate_time']:.1f}x\n"
                )
                f.write(
                    f"**Diarize RTF**: {self.duration/self.stats['diarize_time']:.1f}x\n"
                )

            self.print_stats(self.start_time, wall_time)
            print(f"\nTranscript saved to: {self.transcript_path}")
            print(f"Session log saved to: {self.output_path}")


def list_devices():
    """List available audio input devices."""
    devices = sd.query_devices()
    print("\nAvailable Audio Devices:")
    print("-" * 60)
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(
                f"  [{i}] {dev['name']} (in={dev['max_input_channels']}, sr={dev['default_samplerate']:.0f})"
            )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Real-Time Audio Streaming Capture")
    parser.add_argument(
        "--device", type=int, default=1, help="Audio device ID (default: 1)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Capture duration in seconds (default: 600)",
    )
    parser.add_argument(
        "--chunk",
        type=float,
        default=3.0,
        help="Chunk duration in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio devices"
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # Get device name
    devices = sd.query_devices()
    device_name = (
        devices[args.device]["name"] if args.device < len(devices) else "Unknown"
    )

    capture = RealTimeCapture(
        device=args.device,
        device_name=device_name,
        duration=args.duration,
        chunk_duration=args.chunk,
    )
    capture.run()


if __name__ == "__main__":
    main()
