"""
WhisperTurbo Gradio UI - Production-Ready Interface
Real-time speech translation with speaker diarization.
Supports source language selection and target language translation.
"""

import gradio as gr
import sounddevice as sd
import warnings
import threading
import time
import numpy as np
from typing import Optional, List, Tuple
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# Set HF_TOKEN for pyannote
DEFAULT_HF_TOKEN = ""
if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = DEFAULT_HF_TOKEN

# Language options - Expanded to all major languages
SOURCE_LANGUAGES = [
    ("Auto-detect", "auto"),
    ("Korean", "ko"),
    ("Japanese", "ja"),
    ("Chinese (Simplified)", "zh"),
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Russian", "ru"),
    ("Portuguese", "pt"),
    ("Italian", "it"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
    ("Thai", "th"),
    ("Vietnamese", "vi"),
    ("Indonesian", "id"),
    ("Dutch", "nl"),
    ("Polish", "pl"),
    ("Turkish", "tr"),
    ("Ukrainian", "uk"),
]

TARGET_LANGUAGES = [
    ("English", "en"),
    ("Korean", "ko"),
    ("Japanese", "ja"),
    ("Chinese (Simplified)", "zh"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Russian", "ru"),
    ("Portuguese", "pt"),
    ("Italian", "it"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
    ("Thai", "th"),
    ("Vietnamese", "vi"),
    ("Indonesian", "id"),
    ("Dutch", "nl"),
    ("Polish", "pl"),
    ("Turkish", "tr"),
    ("Ukrainian", "uk"),
]


class GradioGUI:
    def __init__(self, fusion, pipeline=None):
        self.fusion = fusion
        self.pipeline = pipeline
        self._is_running = False
        self._correction_running = False
        self._audio_level = 0.0
        self._segments_buffer = []
        self._segments_lock = threading.Lock()
        self._max_segments = 200
        self._metrics = {"latency": 0.0, "rtf": 0.0, "segments": 0, "speakers": 0}
        self._source_language = "auto"
        self._target_language = "en"

        self.devices = self._get_active_audio_devices()

        self._monitor_thread = None
        self._monitoring = False

    def _get_active_audio_devices(self) -> List[Tuple[str, int]]:
        """Get only active audio input devices, filter out inactive ones."""
        try:
            devices = sd.query_devices()
            device_list = []
            seen_names = set()

            for i in range(len(devices)):
                try:
                    dev = sd.query_devices(i)
                    if dev and dev.get("max_input_channels", 0) > 0:
                        name = dev.get("name", f"Device {i}")
                        ch = dev.get("max_input_channels", 0)
                        sr = dev.get("default_samplerate", 0)

                        if sr not in [44100, 48000]:
                            continue

                        name_lower = name.lower()
                        skip_patterns = ["speaker", "headphone"]
                        is_output_device = (
                            any(p in name_lower for p in skip_patterns)
                            and "cable" not in name_lower
                        )
                        if is_output_device:
                            continue

                        norm_name = name
                        for suffix in [
                            " (Realtek(R) Au",
                            " (Realtek(R) Audio)",
                            " (Realtek HD Audio ",
                            " (Realtek HD ",
                            " (VB-Audio Virtual ",
                            " (VB-Audio Virtual Cable)",
                            " (VB-Audio Point)",
                        ]:
                            if suffix in norm_name:
                                norm_name = norm_name.split(suffix)[0]

                        if norm_name in seen_names:
                            continue
                        seen_names.add(norm_name)

                        label = f"[{i}] {name} ({ch}ch, {sr:.0f}Hz)"
                        device_list.append((label, i, sr))
                except Exception:
                    continue

            if not device_list:
                return [("Default Microphone", 0)]

            def sort_key(x):
                is_cable = 1 if "cable" in x[0].lower() else 0
                is_44k = 1 if x[2] == 44100 else 0
                return (-is_cable, -is_44k, x[0])

            device_list.sort(key=sort_key)

            return [(d[0], d[1]) for d in device_list[:6]]
        except Exception:
            return [("Default Microphone", 0)]

    def _monitor_audio_level(self, device_id: int):
        """Monitor audio level from a specific device."""
        try:
            dev = sd.query_devices(device_id)
            native_sr = int(dev["default_samplerate"])

            with sd.InputStream(
                device=device_id, channels=1, samplerate=native_sr, blocksize=512
            ) as stream:
                while self._monitoring:
                    audio, _ = stream.read(512)
                    self._audio_level = float(np.abs(audio).mean())
                    time.sleep(0.05)
        except Exception:
            pass

    def get_audio_levels(self) -> str:
        """Get audio levels for active devices only."""
        levels = []
        for label, device_id in self.devices[:5]:
            try:
                dev = sd.query_devices(device_id)
                native_sr = int(dev["default_samplerate"])

                with sd.InputStream(
                    device=device_id, channels=1, samplerate=native_sr, blocksize=1600
                ) as stream:
                    audio, _ = stream.read(1600)
                    level = float(np.abs(audio).mean())
                    status = "ACTIVE" if level > 0.001 else "silent"
                    bar_len = min(20, int(level * 500))
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    levels.append(f"{label}: {bar} [{status}]")
            except Exception as e:
                levels.append(f"{label}: ERROR ({e})")
        return "\n".join(levels) if levels else "No devices found"

    def get_interface(self):
        """Create the Gradio interface."""

        def on_start(device_str, source_lang, target_lang):
            if not self.pipeline:
                return "Error: Pipeline not connected"

            try:
                device_id = int(device_str.split("[")[1].split("]")[0])

                # Get language codes
                source_code = dict(SOURCE_LANGUAGES).get(source_lang, "auto")
                target_code = dict(TARGET_LANGUAGES).get(target_lang, "en")

                self._source_language = source_code
                self._target_language = target_code

                # Set languages in pipeline
                if hasattr(self.pipeline, "set_languages"):
                    self.pipeline.set_languages(source_code, target_code)

                self._monitoring = True
                self._monitor_thread = threading.Thread(
                    target=self._monitor_audio_level, args=(device_id,), daemon=True
                )
                self._monitor_thread.start()

                self.pipeline.set_audio_device(device_id)
                self.pipeline.start()
                self._is_running = True
                return f"Recording: {source_lang} → {target_lang}"
            except Exception as e:
                return f"Error: {e}"

        def on_stop():
            if not self.pipeline:
                return "Error: Pipeline not connected"

            try:
                self._monitoring = False
                if self._monitor_thread:
                    self._monitor_thread.join(timeout=1.0)
                self.pipeline.stop()
                self._is_running = False
                return "Recording stopped"
            except Exception as e:
                return f"Error: {e}"

        def refresh_data():
            """Get current transcription data with source and target text."""
            segments = []
            try:
                if self.fusion and hasattr(self.fusion, "_pending_segments"):
                    all_seg = list(self.fusion._pending_segments[-self._max_segments :])

                    for seg in all_seg:
                        time_str = f"{seg.start:.1f}s - {seg.end:.1f}s"
                        speaker = seg.speaker or "Speaker"
                        source = seg.source_text or ""
                        target = seg.target_text or source
                        correction = seg.correction_status or "fast"
                        segments.append([time_str, speaker, source, target, correction])
            except Exception:
                pass

            latency = f"{self._metrics['latency']:.2f}"
            rtf = f"{self._metrics['rtf']:.2f}"
            segments_count = str(len(segments))
            speakers = "0"

            if segments:
                unique_speakers = set(seg[1] for seg in segments)
                speakers = str(len(unique_speakers))

            audio_level = f"{self._audio_level:.4f}"
            correction_status = "Running" if self._correction_running else "Idle"

            return segments, latency, rtf, segments_count, speakers, audio_level, correction_status

        def export_transcript():
            """Export transcript as markdown file."""
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = os.path.join(
                    os.environ.get("USERPROFILE", "."),
                    "Desktop",
                    f"whisperturbo_transcript_{timestamp}.md",
                )

                segments = []
                if self.fusion and hasattr(self.fusion, "_pending_segments"):
                    segments = list(self.fusion._pending_segments)

                with open(export_path, "w", encoding="utf-8") as f:
                    f.write(f"# WhisperTurbo Transcript\n\n")
                    f.write(f"**Exported**: {datetime.now()}\n")
                    f.write(f"**Segments**: {len(segments)}\n\n")
                    f.write(f"| Time | Speaker | Source | Translation | Status |\n")
                    f.write(f"|------|---------|--------|-------------|--------|\n")
                    for seg in segments:
                        time_str = f"{seg.start:.1f}s - {seg.end:.1f}s"
                        speaker = seg.speaker or "Speaker"
                        source = seg.source_text or ""
                        target = seg.target_text or source
                        status = seg.correction_status or "fast"
                        f.write(f"| {time_str} | {speaker} | {source} | {target} | {status} |\n")

                return f"Exported to: {export_path}"
            except Exception as e:
                return f"Export error: {e}"

        def on_generate_summary():
            """Generate summary from current transcription data."""
            summary = self.generate_summary()
            key_points = summary.get("key_points", "No data")
            speaker_contrib = summary.get("speaker_contributions", "No data")
            full_summary = summary.get("full_summary", "No data")
            return key_points, speaker_contrib, full_summary

        def refresh_devices():
            self.devices = self._get_active_audio_devices()
            return gr.update(
                choices=[d[0] for d in self.devices],
                value=self.devices[0][0] if self.devices else None,
            )

        def test_audio_levels():
            return self.get_audio_levels()

        # === UI Layout ===
        with gr.Blocks(
            title="WhisperTurbo - Real-Time Translation",
            css="""
            .gradio-container { max-width: 1400px !important; margin: auto !important; }
            #header { text-align: center; padding: 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-bottom: 20px; }
            #header h1 { color: white !important; margin: 0 !important; font-size: 2em; }
            #header p { color: rgba(255,255,255,0.9) !important; margin: 8px 0 0 0 !important; font-size: 1.1em; }
            """,
        ) as demo:

            gr.HTML(
                """
            <div id="header">
                <h1>WhisperTurbo</h1>
                <p>Real-Time Speech Translation with Speaker Diarization</p>
            </div>
            """
            )

            with gr.Row(equal_height=True):
                # Left panel - Controls
                with gr.Column(scale=1):
                    gr.Markdown("### Audio Input")

                    device_dropdown = gr.Dropdown(
                        choices=[d[0] for d in self.devices],
                        value=self.devices[0][0] if self.devices else None,
                        label="Input Device",
                        info="Select microphone or audio loopback device",
                    )

                    with gr.Row():
                        refresh_devices_btn = gr.Button(
                            "Refresh Devices", size="sm", scale=1
                        )
                        test_levels_btn = gr.Button("Test Levels", size="sm", scale=1)

                    audio_levels_output = gr.Textbox(
                        label="Device Audio Levels",
                        lines=4,
                        interactive=False,
                    )

                    gr.Markdown("### Languages")

                    source_lang_dropdown = gr.Dropdown(
                        choices=[lang[0] for lang in SOURCE_LANGUAGES],
                        value="Auto-detect",
                        label="Source Language",
                        info="Language of the input audio",
                    )

                    target_lang_dropdown = gr.Dropdown(
                        choices=[lang[0] for lang in TARGET_LANGUAGES],
                        value="English",
                        label="Target Language",
                        info="Language to translate to",
                    )

                    with gr.Row():
                        start_btn = gr.Button(
                            "Start Recording", variant="primary", size="lg", scale=1
                        )
                        stop_btn = gr.Button("Stop", variant="stop", size="lg", scale=1)

                    status_text = gr.Textbox(
                        label="Status",
                        value="Ready",
                        interactive=False,
                    )

                # Right panel - Performance
                with gr.Column(scale=1):
                    gr.Markdown("### Performance Metrics")

                    with gr.Row():
                        latency_box = gr.Textbox(
                            label="Latency (s)", value="0.0", scale=1
                        )
                        rtf_box = gr.Textbox(label="RTF (x)", value="0.0", scale=1)

                    with gr.Row():
                        segments_box = gr.Textbox(label="Segments", value="0", scale=1)
                        speakers_box = gr.Textbox(label="Speakers", value="0", scale=1)

                    audio_level_box = gr.Textbox(
                        label="Audio Level", value="0.0000", scale=1
                    )

                    correction_status_box = gr.Textbox(
                        label="Correction", value="Idle", scale=1
                    )

                    gr.Markdown("### Export")
                    export_btn = gr.Button(
                        "Export Transcript", variant="secondary", size="sm"
                    )
                    export_status = gr.Textbox(
                        label="Export", value="", interactive=False, lines=1
                    )

            gr.Markdown("---")

            # Transcription display
            gr.Markdown("### Live Transcription & Translation")

            with gr.Row():
                refresh_btn = gr.Button("Refresh Now", size="sm", scale=1)

            transcription_table = gr.Dataframe(
                headers=["Time", "Speaker", "Transcription", "Translation", "Status"],
                value=[],
                max_height=500,
                wrap=True,
                column_widths=["10%", "8%", "32%", "32%", "18%"],
                interactive=False,
            )

            # === Event Handlers ===
            start_btn.click(
                fn=on_start,
                inputs=[device_dropdown, source_lang_dropdown, target_lang_dropdown],
                outputs=[status_text],
            )
            stop_btn.click(fn=on_stop, outputs=[status_text])
            refresh_btn.click(
                fn=refresh_data,
                outputs=[
                    transcription_table,
                    latency_box,
                    rtf_box,
                    segments_box,
                    speakers_box,
                    audio_level_box,
                    correction_status_box,
                ],
            )
            test_levels_btn.click(fn=test_audio_levels, outputs=[audio_levels_output])
            refresh_devices_btn.click(fn=refresh_devices, outputs=[device_dropdown])
            export_btn.click(fn=export_transcript, outputs=[export_status])

            # Auto-refresh every 0.5s
            demo.load(
                fn=refresh_data,
                outputs=[
                    transcription_table,
                    latency_box,
                    rtf_box,
                    segments_box,
                    speakers_box,
                    audio_level_box,
                    correction_status_box,
                ],
            )

            gr.Timer(0.5).tick(
                fn=refresh_data,
                outputs=[
                    transcription_table,
                    latency_box,
                    rtf_box,
                    segments_box,
                    speakers_box,
                    audio_level_box,
                    correction_status_box,
                ],
            )

            gr.Markdown("---")

            with gr.Row():
                generate_summary_btn = gr.Button(
                    "Generate Summary", variant="secondary", size="sm", scale=1
                )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Key Points")
                    key_points_output = gr.Textbox(
                        lines=6,
                        interactive=False,
                        placeholder="Click 'Generate Summary' after stopping...",
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### Speaker Contributions")
                    speaker_contrib_output = gr.Textbox(
                        lines=6,
                        interactive=False,
                        placeholder="Click 'Generate Summary' after stopping...",
                    )

            gr.Markdown("### Full Summary")
            with gr.Accordion("Show Full Transcript Summary", open=False):
                full_summary_output = gr.Textbox(
                    lines=10,
                    interactive=False,
                    placeholder="Click 'Generate Summary' after stopping...",
                )

            generate_summary_btn.click(
                fn=on_generate_summary,
                outputs=[key_points_output, speaker_contrib_output, full_summary_output],
            )

        return demo

    def stop(self):
        self._is_running = False
        self._monitoring = False

    def update_kpis(self, latency, rtf, processing_rate):
        self._metrics = {
            "latency": latency,
            "rtf": rtf,
            "segments": self._metrics.get("segments", 0) + 1,
            "speakers": self._metrics.get("speakers", 0),
        }

    def set_correction_running(self, is_running: bool) -> None:
        self._correction_running = is_running

    def generate_summary(self) -> dict:
        """Generate summary from transcription segments."""
        segments = []
        try:
            if self.fusion and hasattr(self.fusion, "_pending_segments"):
                segments = list(self.fusion._pending_segments)
        except Exception:
            pass

        if not segments:
            return {
                "key_points": "No transcription data available.",
                "speaker_contributions": "No speakers detected.",
                "full_summary": "No transcription data available.",
            }

        # Key points - extract unique sentences/phrases
        key_points = []
        seen_texts = set()
        for seg in segments:
            text = seg.target_text or seg.source_text
            if text and text not in seen_texts and len(text) > 10:
                seen_texts.add(text)
                key_points.append(f"• {text[:100]}{'...' if len(text) > 100 else ''}")

        if len(key_points) > 5:
            key_points = key_points[:5]

        # Speaker contributions
        speaker_times = {}
        for seg in segments:
            speaker = seg.speaker or "Unknown"
            if speaker not in speaker_times:
                speaker_times[speaker] = {"count": 0, "duration": 0.0}
            speaker_times[speaker]["count"] += 1
            speaker_times[speaker]["duration"] += seg.end - seg.start

        speaker_contributions = []
        for speaker, stats in sorted(speaker_times.items(), key=lambda x: x[1]["duration"], reverse=True):
            duration = stats["duration"]
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            speaker_contributions.append(
                f"• {speaker}: {stats['count']} segments, {minutes}m {seconds}s"
            )

        # Full summary - grouped by speaker
        full_summary_parts = []
        current_speaker = None
        for seg in segments:
            speaker = seg.speaker or "Unknown"
            if speaker != current_speaker:
                if full_summary_parts:
                    full_summary_parts.append("")
                full_summary_parts.append(f"[{speaker}]")
                current_speaker = speaker

            text = seg.target_text or seg.source_text
            if text:
                full_summary_parts.append(f"  {text}")

        return {
            "key_points": "\n".join(key_points) if key_points else "No key points extracted.",
            "speaker_contributions": "\n".join(speaker_contributions) if speaker_contributions else "No speakers detected.",
            "full_summary": "\n".join(full_summary_parts) if full_summary_parts else "No summary available.",
        }

    def add_segments(self, segments):
        pass

    def serve(self, port=7860, **kwargs):
        demo = self.get_interface()
        demo.launch(server_port=port, show_error=True, share=False, **kwargs)
