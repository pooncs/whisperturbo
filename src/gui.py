import logging
import os
import threading
import time
from dataclasses import dataclass

import pandas as pd
import panel as pn

from .config import CONFIG
from .fusion import Fusion, TranslatedSegment

logger = logging.getLogger(__name__)


@dataclass
class KPIMetrics:
    latency: float = 0.0
    rtf: float = 0.0
    total_segments: int = 0
    active_speakers: int = 0
    processing_rate: float = 0.0


class TranslationGUI:
    def __init__(
        self,
        fusion: Fusion,
        pipeline=None,
        title: str = "Real-Time Speech Translation",
    ):
        self.fusion = fusion
        self.pipeline = pipeline
        self.title = title

        self._segments: list[TranslatedSegment] = []
        self._segments_lock = threading.Lock()

        self._kpis = KPIMetrics()
        self._kpis_lock = threading.Lock()

        self._start_time = time.time()
        self._last_processing_time = 0.0
        self._last_audio_duration = 0.0

        self._is_running = False
        self._refresh_task = None
        self.speaker_colors: dict[str, str] = {}
        self._paused = False  # To track local state, but actually use pipeline

        self._setup_gui()

    def _setup_gui(self) -> None:
        pn.extension("tabulator", sizing_mode="stretch_width")

        self._title = pn.pane.Markdown(
            f"## {self.title}\n**Korean -> English Real-Time Translation**",
            sizing_mode="stretch_width",
        )

        self._latency_pane = pn.pane.Markdown("0.00 s", styles={"color": "blue"})
        self._rtf_pane = pn.pane.Markdown("0.00x", styles={"color": "green"})
        self._segments_pane = pn.pane.Markdown("0")
        self._speakers_pane = pn.pane.Markdown("0")
        self._rate_pane = pn.pane.Markdown("0.00 segments/s")

        self._kpi_panel = pn.Column(
            pn.Row(
                pn.pane.Markdown("### KPIs", styles={"font-weight": "bold"}),
            ),
            pn.Row(
                pn.pane.Markdown("**Latency:**", styles={"font-weight": "bold"}),
                self._latency_pane,
                pn.pane.Markdown(" **RTF:**", styles={"font-weight": "bold"}),
                self._rtf_pane,
            ),
            pn.Row(
                pn.pane.Markdown("**Segments:**", styles={"font-weight": "bold"}),
                self._segments_pane,
                pn.pane.Markdown(" **Speakers:**", styles={"font-weight": "bold"}),
                self._speakers_pane,
            ),
            pn.Row(
                pn.pane.Markdown("**Processing Rate:**", styles={"font-weight": "bold"}),
                self._rate_pane,
            ),
            sizing_mode="stretch_width",
        )

        self._table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["Time", "Speaker", "Text"]),
            frozen_columns=["Time"],
            height=400,
            page_size=CONFIG.GUI_MAX_ROWS,
            sizing_mode="stretch_width",
        )

        self._export_csv_btn = pn.widgets.Button(
            name="Export CSV", button_type="primary", width=100
        )
        self._export_jsonl_btn = pn.widgets.Button(
            name="Export JSONL", button_type="primary", width=100
        )
        self._export_srt_btn = pn.widgets.Button(
            name="Export SRT", button_type="primary", width=100
        )
        self._clear_btn = pn.widgets.Button(name="Clear Log", button_type="warning", width=100)

        self._pause_btn = pn.widgets.Button(name="Pause", button_type="warning", width=100)
        self._resume_btn = pn.widgets.Button(name="Resume", button_type="success", width=100)
        self._diarization_toggle = pn.widgets.Checkbox(name="Enable Diarization", value=True)

        self._export_dir_input = pn.widgets.TextInput(
            name="Export Directory", value=os.getcwd(), width=400
        )

        self._speaker_filter = pn.widgets.Select(
            name="Speaker Filter", options=["All"], value="All"
        )

        self._mode_select = pn.widgets.RadioButtonGroup(
            name="Mode",
            options=["Local", "Connect Online"],
            value="Local",
            width=200,
        )
        self._websocket_url_input = pn.widgets.TextInput(
            name="WebSocket URL",
            value=CONFIG.WEBSOCKET_URL,
            width=400,
            placeholder="ws://localhost:8765",
            disabled=True,
        )
        self._connect_btn = pn.widgets.Button(
            name="Connect", button_type="success", width=100, disabled=True
        )
        self._disconnect_btn = pn.widgets.Button(
            name="Disconnect", button_type="danger", width=100, disabled=True
        )
        self._connection_status_pane = pn.pane.Markdown(
            "**Status:** Disconnected", styles={"color": "red"}
        )

        self._websocket_client = None
        self._websocket_thread = None

        self._active_speakers_pane = pn.pane.Markdown("### Active Speakers\n- None")

        self._log_pane = pn.widgets.TextAreaInput(
            name="Logs",
            value="",
            height=100,
            disabled=True,
            sizing_mode="stretch_width",
        )

        self._control_panel = pn.Column(
            pn.pane.Markdown("### Controls", styles={"font-weight": "bold"}),
            pn.Row(
                self._pause_btn,
                self._resume_btn,
                self._diarization_toggle,
            ),
            pn.Row(
                self._clear_btn,
            ),
            pn.Row(
                pn.pane.Markdown("**Export Dir:**", styles={"font-weight": "bold"}),
                self._export_dir_input,
            ),
            pn.Row(
                self._export_csv_btn,
                self._export_jsonl_btn,
                self._export_srt_btn,
            ),
            pn.Row(
                self._speaker_filter,
            ),
            sizing_mode="stretch_width",
        )

        self._websocket_panel = pn.Column(
            pn.pane.Markdown("### WebSocket Connection", styles={"font-weight": "bold"}),
            pn.Row(
                pn.pane.Markdown("**Mode:**", styles={"font-weight": "bold"}),
                self._mode_select,
            ),
            pn.Row(
                pn.pane.Markdown("**WebSocket URL:**", styles={"font-weight": "bold"}),
                self._websocket_url_input,
            ),
            pn.Row(
                self._connect_btn,
                self._disconnect_btn,
            ),
            self._connection_status_pane,
            sizing_mode="stretch_width",
        )

        self._export_csv_btn.on_click(self._on_export_csv)
        self._export_jsonl_btn.on_click(self._on_export_jsonl)
        self._export_srt_btn.on_click(self._on_export_srt)
        self._clear_btn.on_click(self._on_clear)

        self._pause_btn.on_click(self._on_pause)
        self._resume_btn.on_click(self._on_resume)
        self._diarization_toggle.param.watch(self._on_diarization_toggle, "value")
        self._mode_select.param.watch(self._on_mode_change, "value")
        self._connect_btn.on_click(self._on_connect)
        self._disconnect_btn.on_click(self._on_disconnect)

        self._status_pane = pn.pane.Markdown("**Status:** Ready", styles={"color": "gray"})

        self._layout = pn.Column(
            self._title,
            pn.layout.Divider(),
            self._kpi_panel,
            pn.layout.Divider(),
            self._websocket_panel,
            pn.layout.Divider(),
            self._table,
            pn.layout.Divider(),
            self._active_speakers_pane,
            pn.layout.Divider(),
            self._control_panel,
            pn.layout.Divider(),
            self._log_pane,
            pn.layout.Divider(),
            self._status_pane,
            sizing_mode="stretch_width",
        )

    def add_segment(self, segment: TranslatedSegment) -> None:
        with self._segments_lock:
            self._segments.append(segment)

            if len(self._segments) > CONFIG.GUI_MAX_ROWS * 2:
                self._segments = self._segments[-CONFIG.GUI_MAX_ROWS :]

    def add_segments(self, segments: list[TranslatedSegment]) -> None:
        with self._segments_lock:
            self._segments.extend(segments)

            if len(self._segments) > CONFIG.GUI_MAX_ROWS * 2:
                self._segments = self._segments[-CONFIG.GUI_MAX_ROWS :]

    def update_kpis(
        self,
        latency: float,
        rtf: float,
        processing_rate: float,
    ) -> None:
        with self._kpis_lock:
            self._kpis.latency = latency
            self._kpis.rtf = rtf
            self._kpis.processing_rate = processing_rate
            self._kpis.total_segments = len(self._segments)

            speakers = {s.speaker for s in self._segments if s.speaker}
            self._kpis.active_speakers = len(speakers)

    def _refresh_table(self) -> None:
        with self._segments_lock:
            if not self._segments:
                return

            recent_segments = self._segments[-CONFIG.GUI_MAX_ROWS :]

            data = []
            for seg in recent_segments:
                time_str = f"{seg.start:.2f}s - {seg.end:.2f}s"
                speaker = seg.speaker or "UNKNOWN"
                text = seg.target_text or seg.source_text

                # Assign colors to speakers
                if speaker not in self.speaker_colors:
                    color_idx = len(self.speaker_colors) % len(CONFIG.SPEAKER_COLORS)
                    self.speaker_colors[speaker] = CONFIG.SPEAKER_COLORS[color_idx]

                color = self.speaker_colors[speaker]

                data.append(
                    {
                        "Time": time_str,
                        "Speaker": f'<span style="color: {color}; font-weight: bold;">{speaker}</span>',
                        "Text": text,
                    }
                )

        # Update speaker filter options
        speakers = sorted({s.speaker for s in recent_segments if s.speaker})
        current_options = list(self._speaker_filter.options)
        if speakers != current_options[1:]:  # Skip "All"
            self._speaker_filter.options = ["All"] + speakers

        # Filter data if needed
        if self._speaker_filter.value != "All":
            filtered_data = [d for d in data if d["Speaker"] == self._speaker_filter.value]
        else:
            filtered_data = data

        # Assign colors to speakers
        for seg in recent_segments:
            if seg.speaker and seg.speaker not in self.speaker_colors:
                color_idx = len(self.speaker_colors) % len(CONFIG.SPEAKER_COLORS)
                self.speaker_colors[seg.speaker] = CONFIG.SPEAKER_COLORS[color_idx]
            # For now, don't color text, just store

        df = pd.DataFrame(filtered_data)

        self._table.value = df

        # Update active speakers pane
        active = [f"- {s}" for s in speakers]
        self._active_speakers_pane.object = "### Active Speakers\n" + (
            "\n".join(active) if active else "- None"
        )

        with self._kpis_lock:
            self._latency_pane.object = f"{self._kpis.latency:.2f} s"
            self._rtf_pane.object = f"{self._kpis.rtf:.2f}x"
            self._segments_pane.object = str(self._kpis.total_segments)
            self._speakers_pane.object = str(self._kpis.active_speakers)
            self._rate_pane.object = f"{self._kpis.processing_rate:.2f} segments/s"

    def _on_export_csv(self, event) -> None:
        try:
            import os

            dir_path = self._export_dir_input.value
            os.makedirs(dir_path, exist_ok=True)
            filepath = os.path.join(dir_path, f"transcription_{int(time.time())}.csv")
            metadata = self._get_metadata()
            self.fusion.export_csv_with_metadata(filepath, metadata)
            self._status_pane.object = f"**Status:** Exported CSV to {filepath}"
            self._append_log(f"Exported CSV to {filepath}")
        except Exception as e:
            logger.error(f"Export CSV error: {e}")
            self._status_pane.object = f"**Status:** Export failed - {e}"
            self._append_log(f"Export CSV failed: {e}")

    def _on_export_jsonl(self, event) -> None:
        try:
            import os

            dir_path = self._export_dir_input.value
            os.makedirs(dir_path, exist_ok=True)
            filepath = os.path.join(dir_path, f"transcription_{int(time.time())}.jsonl")
            metadata = self._get_metadata()
            self.fusion.export_jsonl_with_metadata(filepath, metadata)
            self._status_pane.object = f"**Status:** Exported JSONL to {filepath}"
            self._append_log(f"Exported JSONL to {filepath}")
        except Exception as e:
            logger.error(f"Export JSONL error: {e}")
            self._status_pane.object = f"**Status:** Export failed - {e}"
            self._append_log(f"Export JSONL failed: {e}")

    def _on_export_srt(self, event) -> None:
        try:
            import os

            dir_path = self._export_dir_input.value
            os.makedirs(dir_path, exist_ok=True)
            filepath = os.path.join(dir_path, f"transcription_{int(time.time())}.srt")
            metadata = self._get_metadata()
            self.fusion.export_srt_with_metadata(filepath, metadata)
            self._status_pane.object = f"**Status:** Exported SRT to {filepath}"
            self._append_log(f"Exported SRT to {filepath}")
        except Exception as e:
            logger.error(f"Export SRT error: {e}")
            self._status_pane.object = f"**Status:** Export failed - {e}"
            self._append_log(f"Export SRT failed: {e}")

    def _on_clear(self, event) -> None:
        with self._segments_lock:
            self._segments.clear()
        self.fusion.clear()
        self._table.value = pd.DataFrame(columns=["Time", "Speaker", "Text"])
        self._status_pane.object = "**Status:** Log cleared"

    def _on_pause(self, event) -> None:
        if self.pipeline:
            self.pipeline.pause()
        self._paused = True
        self._status_pane.object = "**Status:** Pipeline paused"
        self._append_log("Pipeline paused")

    def _on_resume(self, event) -> None:
        if self.pipeline:
            self.pipeline.resume()
        self._paused = False
        self._status_pane.object = "**Status:** Pipeline resumed"
        self._append_log("Pipeline resumed")

    def _on_diarization_toggle(self, event) -> None:
        if self.pipeline:
            self.pipeline.enable_diarization = event.new
        self._append_log(f"Diarization {'enabled' if event.new else 'disabled'}")

    def _append_log(self, message: str) -> None:
        current = self._log_pane.value or ""
        timestamp = f"{time.time():.2f}"
        self._log_pane.value = f"{current}{timestamp}: {message}\n"

    def _get_metadata(self) -> dict:
        return {
            "start_time": self._start_time,
            "whisper_model": CONFIG.WHISPER_MODEL,
            "gui_refresh_rate": CONFIG.GUI_REFRESH_RATE,
            "websocket_mode": CONFIG.WEBSOCKET_MODE,
            "websocket_url": CONFIG.WEBSOCKET_URL,
            "export_time": time.time(),
        }

    def _on_mode_change(self, event) -> None:
        if event.new == "Connect Online":
            self._websocket_url_input.disabled = False
            self._connect_btn.disabled = False
            self._disconnect_btn.disabled = True
            CONFIG.WEBSOCKET_MODE = True
            self._connection_status_pane.object = "**Status:** Ready to connect"
            self._connection_status_pane.styles = {"color": "orange"}
        else:
            self._websocket_url_input.disabled = True
            self._connect_btn.disabled = True
            self._disconnect_btn.disabled = True
            if self._websocket_client:
                self._disconnect_websocket()
            CONFIG.WEBSOCKET_MODE = False
            self._connection_status_pane.object = "**Status:** Disconnected"
            self._connection_status_pane.styles = {"color": "red"}
        self._append_log(f"Mode changed to {event.new}")

    def _on_connect(self, event) -> None:
        url = self._websocket_url_input.value or f"ws://localhost:{CONFIG.WEBSOCKET_PORT}"
        if not url.startswith("ws://"):
            url = f"ws://{url}"
        CONFIG.WEBSOCKET_URL = url
        self._connect_websocket(url)

    def _on_disconnect(self, event) -> None:
        self._disconnect_websocket()

    def _connect_websocket(self, url: str) -> None:
        try:
            import websocket

            self._websocket_client = websocket.WebSocketApp(
                url,
                on_message=self._on_websocket_message,
                on_error=self._on_websocket_error,
                on_open=self._on_websocket_open,
                on_close=self._on_websocket_close,
            )
            self._websocket_thread = threading.Thread(
                target=self._websocket_client.run_forever, daemon=True
            )
            self._websocket_thread.start()
            self._connection_status_pane.object = f"**Status:** Connecting to {url}..."
            self._connection_status_pane.styles = {"color": "orange"}
            self._append_log(f"Connecting to WebSocket: {url}")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self._connection_status_pane.object = f"**Status:** Connection failed - {e}"
            self._connection_status_pane.styles = {"color": "red"}
            self._append_log(f"WebSocket connection failed: {e}")

    def _disconnect_websocket(self) -> None:
        if self._websocket_client:
            self._websocket_client.close()
            self._websocket_client = None
            self._websocket_thread = None
            self._connection_status_pane.object = "**Status:** Disconnected"
            self._connection_status_pane.styles = {"color": "red"}
            self._connect_btn.disabled = False
            self._disconnect_btn.disabled = True
            self._append_log("Disconnected from WebSocket")

    def _on_websocket_message(self, ws, message) -> None:
        try:
            import json

            data = json.loads(message)
            if "text" in data and "start" in data and "end" in data:
                from .fusion import TranslatedSegment

                segment = TranslatedSegment(
                    start=data["start"],
                    end=data["end"],
                    source_text=data.get("source_text", ""),
                    target_text=data.get("text", data.get("source_text", "")),
                    source_language=data.get("source_language", ""),
                    target_language=data.get("target_language", "en"),
                    speaker=data.get("speaker", "UNKNOWN"),
                )
                self.add_segment(segment)
                self._append_log(f"Received: {data['text'][:50]}...")
        except Exception as e:
            logger.error(f"WebSocket message error: {e}")

    def _on_websocket_error(self, ws, error) -> None:
        logger.error(f"WebSocket error: {error}")
        self._connection_status_pane.object = f"**Status:** Error - {error}"
        self._connection_status_pane.styles = {"color": "red"}
        self._append_log(f"WebSocket error: {error}")

    def _on_websocket_open(self, ws) -> None:
        self._connection_status_pane.object = "**Status:** Connected"
        self._connection_status_pane.styles = {"color": "green"}
        self._connect_btn.disabled = True
        self._disconnect_btn.disabled = False
        self._append_log("WebSocket connected")

    def _on_websocket_close(self, ws, close_status_code, close_msg) -> None:
        self._connection_status_pane.object = "**Status:** Disconnected"
        self._connection_status_pane.styles = {"color": "red"}
        self._connect_btn.disabled = False
        self._disconnect_btn.disabled = True
        self._append_log(f"WebSocket closed: {close_status_code} - {close_msg}")

    def show(self, port: int = 5006, show: bool = True) -> None:
        self._is_running = True
        pn.serve(
            self._layout,
            port=port,
            show=show,
            title=self.title,
        )

    def serve(self, port: int = 5006) -> None:
        self._is_running = True
        pn.serve(
            self._layout,
            port=port,
            show=False,
            title=self.title,
            threaded=True,
        )
        pn.state.add_periodic_callback(self._refresh_table, period=CONFIG.GUI_REFRESH_RATE / 1000)
        logger.info(f"GUI server started on port {port}")

    def get_layout(self):
        return self._layout

    def stop(self) -> None:
        self._is_running = False
        logger.info("GUI stopped")
