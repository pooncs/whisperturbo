#!/usr/bin/env python3
import os
import sys
import time
import signal
import argparse
import threading
import webbrowser
from pathlib import Path

try:
    from rich.console import Console
    RICH_AVAILABLE = True
except:
    RICH_AVAILABLE = False

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
VERSION = "1.0.0"

class Colors:
    RESET = chr(27) + "[0m"
    RED = chr(27) + "[91m"
    GREEN = chr(27) + "[92m"
    YELLOW = chr(27) + "[93m"
    BLUE = chr(27) + "[94m"
    CYAN = chr(27) + "[96m"
    BOLD = chr(27) + "[1m"

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

def check_dependencies():
    print_status("Checking dependencies...")
    required = {"faster_whisper": "faster-whisper", "sounddevice": "sounddevice", "torch": "torch", "panel": "panel", "pandas": "pandas"}
    missing = []
    for m, p in required.items():
        try:
            __import__(m)
        except:
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
        print_status("HF_TOKEN: Configured")
        return True
    print_warning("HF_TOKEN not set")
    return False

def start_browser(url, delay=3.0):
    def _open():
        time.sleep(delay)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()

class WhisperTurboLauncher:
    def __init__(self, enable_gui=True, enable_diarization=True, gui_port=5006, open_browser=True, check_models_first=True):
        self.enable_gui = enable_gui
        self.enable_diarization = enable_diarization
        self.gui_port = gui_port
        self.open_browser = open_browser
        self.check_models_first = check_models_first
        self.pipeline = None
        self._shutdown_event = threading.Event()

    def validate_environment(self):
        print(chr(10) + CYAN + "=" * 60 + chr(10) + " Environment Validation " + chr(10) + "=" * 60 + RESET + chr(10))
        ok = True
        if not check_python_version():
            ok = False
        if not check_dependencies():
            ok = False
        if self.check_models_first and not check_models():
            print_warning("Models not found")
        if not self.enable_diarization:
            print_info("Diarization disabled")
        elif not check_hf_token():
            print_warning("HF_TOKEN needed")
            self.enable_diarization = False
        return ok

    def load_pipeline(self):
        print(chr(10) + CYAN + "=" * 60 + chr(10) + " Loading Pipeline " + chr(10) + "=" * 60 + RESET + chr(10))
        try:
            from main import TranslationPipeline
            print_status("Initializing...")
            self.pipeline = TranslationPipeline(enable_gui=self.enable_gui, enable_diarization=self.enable_diarization, gui_port=self.gui_port)
            print_status("Pipeline initialized", True, 2)
            print_status("Loading Whisper model...")
            self.pipeline._whisper.load_model()
            print_status("Whisper loaded", True, 2)
            if self.enable_diarization and self.pipeline._diarization:
                print_status("Loading diarization...")
                self.pipeline._diarization.load_pipeline()
                print_status("Diarization loaded", True, 2)
            print_status("Starting audio...")
            self.pipeline._audio_input.start()
            print_status("Audio started", True, 2)
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
        print(chr(10) + GREEN + BOLD + chr(10) + "=" * 60 + chr(10) + " WhisperTurbo running! " + chr(10) + "=" * 60 + RESET + chr(10))
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
        except:
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
    p.add_argument("--port", type=int, default=5006)
    p.add_argument("--open-browser", action="store_true", default=True)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--skip-models-check", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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
