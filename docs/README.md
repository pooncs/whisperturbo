# WhisperTurbo Documentation

Welcome to the WhisperTurbo documentation. This directory contains comprehensive guides for the WhisperTurbo real-time speech translation system.

## Documentation Structure

| File | Description |
|------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and component design |
| [CONFIGURATION.md](CONFIGURATION.md) | Configuration options and settings |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and solutions |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token (required for speaker diarization)
set HF_TOKEN=your_huggingface_token

# Run using the launcher (recommended)
python launcher.py

# Or run directly
python main.py
```

## Features

- **Real-time Speech Translation**: Korean → English translation using Faster-Whisper
- **Speaker Diarization**: Identify multiple speakers using pyannote.audio
- **Web GUI**: Real-time visualization with Panel
- **Export**: CSV, JSONL, and SRT export formats
- **Low Latency**: < 1 second typical latency
- **Context Carry**: Maintains context across segments for better translation
- **Text Post-processing**: Repetition trimming and whitespace normalization

## Requirements

- Windows 10/11 with NVIDIA GPU (RTX 4090 recommended)
- CUDA 11.8 or 12.x
- Python 3.10+

## Project Structure

```
whisperturbo/
├── main.py              # Main entry point
├── launcher.py          # Environment validation and launcher
├── gui.py               # Alternative GUI entry point
├── download_models.py   # Model download utility
├── build.py             # Build utilities
├── runtime_hook.py      # Runtime hooks
├── init_whisperturbo.py # Initialization script
├── src/
│   ├── __init__.py
│   ├── audio_input.py   # Audio capture from microphone
│   ├── whisper_asr.py   # Speech recognition and translation
│   ├── diarization.py   # Speaker diarization
│   ├── fusion.py        # Speaker-translation merging
│   ├── postprocess.py   # Text post-processing utilities
│   ├── gui.py           # GUI application
│   └── config.py        # Configuration settings
├── tests/               # Unit tests
│   ├── conftest.py
│   ├── test_audio_input.py
│   ├── test_config.py
│   ├── test_diarization.py
│   ├── test_fusion.py
│   ├── test_gui.py
│   ├── test_postprocess.py
│   ├── test_translation_pipeline.py
│   └── test_whisper_asr.py
└── docs/                # Documentation
    ├── README.md
    ├── ARCHITECTURE.md
    ├── CONFIGURATION.md
    └── TROUBLESHOOTING.md
```

## Entry Points

### launcher.py (Recommended)
The launcher provides:
- Environment validation (Python version, CUDA, audio devices)
- Dependency checking
- Model verification
- Automatic browser opening
- Colored console output

### main.py
Direct entry point with full control over all options.

### gui.py
Standalone GUI entry point.

For more details, see the individual documentation files in this directory.
