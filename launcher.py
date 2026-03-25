#!/usr/bin/env python3
import argparse
import os
import signal
import sys
import threading
import time
import webbrowser
from pathlib import Path

try:
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
VERSION = "1.0.0"

# Set default HF_TOKEN if not in environment (for development/testing)
DEFAULT_HF_TOKEN = ""
if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = DEFAULT_HF_TOKEN


class Colors:
    RESET = chr(27) + "[0m"
    RED = chr(27) + "[91m"
    GREEN = chr(27) + "[92m"
    YELLOW = chr(27) + "[93m"
    BLUE = chr(27) + "[94m"
    CYAN = chr(27) + "[96m"
    BOLD = chr(27) + "[1m"


# Global color constants
RESET = Colors.RESET
RED = Colors.RED
GREEN = Colors.GREEN
YELLOW = Colors.YELLOW
BLUE = Colors.BLUE
CYAN = Colors.CYAN
BOLD = Colors.BOLD


def print_banner():
    print(CYAN + BOLD + "WhisperTurbo v" + VERSION + RESET)


def print_status(msg, success=True, indent=0):
    prefix = GREEN + "[+]" + RESET if success else RED + "[-]" + RESET
    print(" " * indent + prefix + " " + msg)


def print_info(msg, indent=0):
    print(" " * indent + BLUE + "[i]" + RESET + " " + msg)


def print_warning(msg, indent=0):
    print(" " * indent + YELLOW + "[!]" + RESET + " " + msg)


def check_python_version():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        print_status("Python 3.9+ required", False)
        return False
    print_status("Python " + str(v.major) + "." + str(v.minor) + "." + str(v.micro))
    return True


def check_cuda():
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_props = []
            for i in range(device_count):
                prop = torch.cuda.get_device_properties(i)
                device_props.append(f"{prop.name} ({prop.total_memory // (1024**3)}GB)")
            device_info = ", ".join(device_props)
            print_status(
                f"CUDA available: {device_count} device(s) - {device_info}", True, 2
            )
            return True
        else:
            print_warning("CUDA not available - using CPU")
            print_info("To enable CUDA support (for NVIDIA GPUs):", 4)
            print_info("1. Ensure NVIDIA drivers are installed (525+)", 6)
            print_info("2. Install CUDA 11.8 or 12.x", 6)
            print_info("3. For CUDA 11.8, run:", 6)
            print_info(
                "   pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118",
                9,
            )
            return False
    except Exception:
        print_warning("PyTorch not available for CUDA check")
        return False


def check_audio_devices():
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
        output_devices = [d for d in devices if d.get("max_output_channels", 0) > 0]

        if input_devices:
            default_input = sd.default.device[0]
            default_name = (
                devices[default_input]["name"]
                if default_input < len(devices)
                else "Unknown"
            )
            print_status(
                f"Audio input: {len(input_devices)} device(s) - default: {default_name}",
                True,
                2,
            )
        else:
            print_warning("No audio input devices found")

        if output_devices:
            default_output = sd.default.device[1]
            default_name = (
                devices[default_output]["name"]
                if default_output < len(devices)
                else "Unknown"
            )
            print_status(
                f"Audio output: {len(output_devices)} device(s) - default: {default_name}",
                True,
                2,
            )
        else:
            print_warning("No audio output devices found")

        return len(input_devices) > 0
    except Exception as e:
        print_status(f"Audio device check failed: {e}", False, 2)
        return False


def check_dependencies():
    print_status("Checking dependencies...")
    required = {
        "faster_whisper": "faster-whisper",
        "sounddevice": "sounddevice",
        "torch": "torch",
        "panel": "panel",
        "pandas": "pandas",
    }
    missing = []
    for m, p in required.items():
        try:
            __import__(m)
        except Exception:
            missing.append(p)
    if missing:
        print_status("Missing: " + ",".join(missing), False)
        return False
    print_status("All dependencies installed", True, 2)
    return True


def check_models():
    print_status("Checking models...")
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        ctranslate_dir = base / "ctranslate2" / "models"
    else:
        ctranslate_dir = Path.home() / ".cache" / "ctranslate2" / "models"
    if ctranslate_dir.exists() and any(ctranslate_dir.iterdir()):
        print_status("Whisper: " + str(ctranslate_dir), True, 2)
    else:
        print_status("Whisper model: Not found", False, 2)
        return False
    return True


def check_hf_token():
    if os.environ.get("HF_TOKEN"):
        print_status("HF_TOKEN: Configured", True, 2)
        return True
    elif os.environ.get("HUGGINGFACE_TOKEN"):
        print_warning("HUGGINGFACE_TOKEN found, but HF_TOKEN is expected")
        print_warning("Consider setting HF_TOKEN instead")
        return False
    else:
        print_warning("HF_TOKEN not set - diarization will be disabled")
        print_info("Get HF token from: https://huggingface.co/settings/tokens", 4)
        return False


def start_browser(url, delay=3.0):
    def _open():
        time.sleep(delay)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()


class WhisperTurboLauncher:
    def __init__(
        self,
        enable_gui=True,
        enable_diarization=True,
        gui_port=7860,
        open_browser=True,
        check_models_first=True,
    ):
        self.enable_gui = enable_gui
        self.enable_diarization = enable_diarization
        self.gui_port = gui_port
        self.open_browser = open_browser
        self.check_models_first = check_models_first
        self.pipeline = None
        self._shutdown_event = threading.Event()

    def validate_environment(self):
        print(
            chr(10)
            + CYAN
            + "=" * 60
            + chr(10)
            + " Environment Validation "
            + chr(10)
            + "=" * 60
            + RESET
            + chr(10)
        )
        ok = True
        if not check_python_version():
            ok = False
        if not check_dependencies():
            ok = False
        check_cuda()
        if not check_audio_devices():
            ok = False
            print_warning("Audio devices required for operation")
        if self.check_models_first and not check_models():
            print_warning("Models not found - run python download_models.py")
        if not self.enable_diarization:
            print_info("Diarization disabled")
        elif not check_hf_token():
            print_warning("HF_TOKEN needed - diarization disabled")
            self.enable_diarization = False
        return ok

    def load_pipeline(self):
        print(
            chr(10)
            + CYAN
            + "=" * 60
            + chr(10)
            + " Loading Pipeline "
            + chr(10)
            + "=" * 60
            + RESET
            + chr(10)
        )
        try:
            from main import TranslationPipeline

            print_status("Initializing...")
            self.pipeline = TranslationPipeline(
                enable_gui=self.enable_gui,
                enable_diarization=self.enable_diarization,
                gui_port=self.gui_port,
            )
            print_status("Pipeline initialized", True, 2)
            print_status("Loading Whisper model...")
            self.pipeline._whisper.load_model()
            print_status("Whisper loaded", True, 2)
            if self.enable_diarization and self.pipeline._diarization:
                print_status("Loading diarization...")
                self.pipeline._diarization.load_pipeline()
                print_status("Diarization loaded", True, 2)
            # Don't auto-start audio - user will click Start Translation in GUI
            # print_status("Starting audio...")
            # self.pipeline._audio_input.start()
            # print_status("Audio started", True, 2)
            if self.enable_gui:
                print_status("Starting GUI on port " + str(self.gui_port) + "...")
                self.pipeline._gui.serve(port=self.gui_port)
                print_status("GUI started", True, 2)
                if self.open_browser:
                    url = "http://localhost:" + str(self.gui_port)
                    print_info("GUI: " + url)
                    start_browser(url)
            return True
        except Exception as e:
            print_status("Error: " + str(e), False)
            import traceback

            traceback.print_exc()
            return False

    def run(self):
        print_banner()
        if not self.validate_environment():
            return 1
        if not self.load_pipeline():
            return 1
        print(
            chr(10)
            + GREEN
            + BOLD
            + "=" * 60
            + chr(10)
            + " WhisperTurbo running! "
            + chr(10)
            + "=" * 60
            + RESET
            + chr(10)
        )
        if self.enable_gui:
            print_info("http://localhost:" + str(self.gui_port))
        print_info("Ctrl+C to stop")

        def sh(s, f):
            print(chr(10) + "Shutting...")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, sh)
        signal.signal(signal.SIGTERM, sh)
        try:
            while not self._shutdown_event.is_set():
                time.sleep(0.5)
        except Exception:
            self.shutdown()
        return 0

    def shutdown(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
                print_status("Stopped")
            except Exception as e:
                print_warning("Shutdown error: " + str(e))


def parse_args():
    p = argparse.ArgumentParser(description="WhisperTurbo Launcher")
    p.add_argument("--no-gui", action="store_true")
    p.add_argument("--no-diarization", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--open-browser", action="store_true", default=True)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--skip-models-check", action="store_true")
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p.parse_args()


def main():
    a = parse_args()
    launcher = WhisperTurboLauncher(
        enable_gui=not a.no_gui,
        enable_diarization=not a.no_diarization,
        gui_port=a.port,
        open_browser=a.open_browser and not a.no_browser,
        check_models_first=not a.skip_models_check,
    )
    import logging

    logging.getLogger().setLevel(getattr(logging, a.log_level))
    return launcher.run()


if __name__ == "__main__":
    sys.exit(main())
