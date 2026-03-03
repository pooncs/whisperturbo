# WhisperTurbo - Real-Time Speech Translation System

Real-time Korean → English speech translation system using Faster-Whisper with speaker diarization.

## Requirements

- Windows 10/11 with NVIDIA GPU (RTX 4090 recommended)
- CUDA 11.8 or 12.x
- Python 3.10+

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd whisperturbo
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
   - For **CUDA 12.x** (default):
     ```bash
     pip install -r requirements.txt
     ```
   - For **CUDA 11.8**:
     ```bash
     pip install -r requirements.txt
     pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
     ```

4. Set HuggingFace token for speaker diarization:
```bash
set HF_TOKEN=your_huggingface_token
```

To get a HuggingFace token:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" permissions
3. Use this token for HF_TOKEN

## Usage

### Basic Usage

```bash
python main.py
```

This will start the GUI on port 5006 and begin real-time translation.

### Command-Line Options

```bash
python main.py --help

Options:
  --no-gui           Disable GUI (headless mode)
  --no-diarization   Disable speaker diarization
  --gui-port PORT    GUI server port (default: 5006)
  --log-level LEVEL  Logging level (DEBUG, INFO, WARNING, ERROR)
```

### Examples

Run with GUI:
```bash
python main.py --gui-port 5006
```

Run without GUI (headless):
```bash
python main.py --no-gui
```

Run without speaker diarization:
```bash
python main.py --no-diarization
```

## GUI Features

- Real-time scrolling table showing:
  - Timestamp
  - Speaker label
  - Translated text
- KPI displays:
  - Latency
  - RTF (Real-Time Factor)
  - Segment count
  - Active speakers
- Export options:
  - CSV
  - JSONL
  - SRT

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│  Microphone    │────>│   AudioInput     │
│  (sounddevice) │     │   (Ring Buffer)  │
└─────────────────┘     └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    v                         v
          ┌─────────────────┐     ┌──────────────────┐
          │  Whisper-Turbo  │     │   Diarization    │
          │  (ASR+Translate)│     │  (pyannote.audio)│
          └────────┬─────────┘     └────────┬─────────┘
                   │                        │
                   └────────────┬───────────┘
                                │
                                v
                       ┌─────────────────┐
                       │     Fusion     │
                       │ (Speaker Map)  │
                       └────────┬────────┘
                                │
                                v
                       ┌─────────────────┐
                       │   Panel GUI     │
                       │  (Real-time)    │
                       └─────────────────┘
```

## Configuration

Edit `src/config.py` to customize:

- Sample rate (default: 16000 Hz)
- Whisper model and parameters
- Diarization settings
- GUI refresh rate

## Model Details

- **ASR Model**: `deepdml/faster-whisper-large-v3-turbo-ct2`
  - Task: translate (Korean → English)
  - Device: CUDA
  - Compute type: float16

- **Diarization Model**: `pyannote/speaker-diarization-community-1`
  - Window size: 15 seconds
  - Overlap: 5 seconds

## Performance

- Typical latency: < 1 second
- Diarization coverage: ≥ 90%

## Troubleshooting

### No audio input
- Check microphone permissions
- Verify sounddevice is working: `python -c "import sounddevice; print(sounddevice.query_devices())"`

### CUDA not available
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Diarization fails
- Ensure HF_TOKEN is set correctly
- Verify access to pyannote models

## License

MIT License
