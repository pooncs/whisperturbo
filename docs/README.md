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

# Run the application
python main.py
```

## Features

- **Real-time Speech Translation**: Korean → English translation using Faster-Whisper
- **Speaker Diarization**: Identify multiple speakers using pyannote.audio
- **Web GUI**: Real-time visualization with Panel
- **Export**: CSV, JSONL, and SRT export formats
- **Low Latency**: < 1 second typical latency

## Requirements

- Windows 10/11 with NVIDIA GPU (RTX 4090 recommended)
- CUDA 12.x
- Python 3.10+

## Project Structure

```
whisperturbo/
├── main.py              # Main entry point
├── gui.py               # GUI application
├── src/
│   ├── audio_input.py   # Audio capture
│   ├── whisper_asr.py   # Speech recognition
│   ├── diarization.py  # Speaker diarization
│   ├── fusion.py       # Speaker-translation merge
│   └── config.py       # Configuration
├── tests/               # Unit tests
└── docs/               # Documentation
```

For more details, see the individual documentation files in this directory.
