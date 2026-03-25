# WhisperTurbo

Real-time Korean to English speech translation with speaker diarization. Captures audio from microphone or system audio loopback, transcribes Korean speech, translates to English, and identifies speakers in real-time.

## Features

- **Real-time transcription**: Whisper large-v3-turbo via CTranslate2 (30x realtime)
- **Real-time translation**: Korean to English translation (31x realtime)
- **Speaker diarization**: pyannote speaker diarization (2.5x realtime)
- **Audio routing**: VB-Audio Virtual Cable for system audio capture
- **Web UI**: Gradio interface for live control and export
- **Continuous streaming**: 10-minute continuous capture with live transcript output

## Requirements

- Windows 10/11 with NVIDIA GPU (CUDA 12.8 support)
- Miniconda or Anaconda
- Python 3.11+

## Installation

### 1. Install Miniconda

Download and install from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

### 2. Create conda environment

```bash
conda create -n whisper python=3.11 -y
conda activate whisper
```

### 3. Install PyTorch (CUDA 12.8)

```bash
pip install torch==2.10.0+cu128 torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Install VB-Audio Virtual Cable (optional)

Download from [https://vb-audio.com/Cable/](https://vb-audio.com/Cable/) and install "Cable in 16ch". This enables capturing system audio (movies, meetings, etc.).

### 6. Set HuggingFace token (for diarization)

The token is hardcoded in `launcher.py` for development. For production, set the `HF_TOKEN` environment variable:

```bash
set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

To get a token:
1. Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Accept the pyannote speaker-diarization-community-1 user conditions at [https://huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

## Usage

### Quick Start (Gradio UI)

```bash
conda activate whisper
python launcher.py
```

Opens at [http://localhost:7860](http://localhost:7860)

### Direct Gradio Launch

```bash
conda activate whisper
python main.py
```

Opens at [http://localhost:5006](http://localhost:5006)

### Real-time Streaming Capture (10 minutes)

```bash
conda activate whisper
python realtime_capture.py --device 1 --duration 600
```

Options:
- `--device N`: Audio device ID (default: 1 = VB-Audio Cable)
- `--duration S`: Capture duration in seconds (default: 600 = 10 min)
- `--chunk S`: Processing chunk duration (default: 3.0)
- `--list-devices`: List available audio devices

### One-shot Capture (60 seconds)

```bash
conda activate whisper
python capture_cable.py
```

## Audio Devices

### Microphone
Use your system microphone directly. Select the microphone device in the UI.

### System Audio (Loopback)
1. Install VB-Audio Virtual Cable
2. Set Windows audio output to "CABLE Input"
3. Select "CABLE Output" device in WhisperTurbo
4. Play Korean audio (movie, meeting, etc.)

## Output Files

All outputs are saved to `C:\Users\hmgics\projects\test_outputs\`:

| File | Description |
|------|-------------|
| `realtime_YYYYMMDD_HHMMSS.md` | Session log with metadata |
| `realtime_YYYYMMDD_HHMMSS_transcript.md` | Timestamped transcript |
| `cable_YYYYMMDD_HHMMSS.wav` | Recorded audio |
| `cable_YYYYMMDD_HHMMSS_korean.txt` | Korean transcription |
| `cable_YYYYMMDD_HHMMSS.md` | Translation output |

## Architecture

```
whisperturbo/
├── main.py              # Pipeline entry point & Gradio launcher
├── launcher.py          # Environment validation & launch
├── gradio_gui.py        # Gradio web interface
├── realtime_capture.py  # 10-minute continuous streaming capture
├── capture_cable.py     # One-shot VB-Audio Cable capture
├── src/
│   ├── config.py        # Configuration & environment variables
│   ├── audio_input.py   # Audio input handler (sounddevice)
│   ├── whisper_asr.py   # Whisper ASR wrapper (faster-whisper)
│   ├── diarization.py   # Speaker diarization (pyannote)
│   ├── fusion.py        # ASR + diarization fusion
│   └── postprocess.py   # Post-processing utilities
├── plan.md              # Project roadmap & phases
└── README.md            # This file
```

## GUI Features

- **Device Selection**: Choose from all available audio input devices
- **Real-time Display**: Live transcription with speaker labels
- **Performance Metrics**: Latency, RTF, segment count, speaker count
- **Audio Level Indicator**: Visual feedback on audio input levels
- **Export**: Download transcript as markdown file
- **File Processing**: Upload audio files for batch translation
- **Auto-refresh**: Updates every 0.5 seconds

## Command-Line Options

### main.py
```bash
python main.py --help
Options:
  --no-gui           Disable GUI (headless mode)
  --no-diarization   Disable speaker diarization
  --gui-port PORT    GUI server port (default: 5006)
  --benchmark        Run in benchmark mode (more verbose metrics)
  --log-level LEVEL  Logging level (DEBUG, INFO, WARNING, ERROR)
```

### launcher.py
```bash
python launcher.py --help
Options:
  --no-gui           Disable GUI
  --no-diarization   Disable speaker diarization
  --port PORT        GUI server port (default: 5006)
  --open-browser     Open browser automatically (default)
  --no-browser       Don't open browser
  --skip-models-check Skip model validation
  --log-level LEVEL  Logging level
```

### realtime_capture.py
```bash
python realtime_capture.py --help
Options:
  --device N         Audio device ID (default: 1)
  --duration S       Capture duration in seconds (default: 600)
  --chunk S          Processing chunk duration (default: 3.0)
  --list-devices     List available audio devices
```

## Troubleshooting

### CUDA not available

```
WARNING: CUDA not available - using CPU
```

**Solution**: Install CUDA-enabled PyTorch:
```bash
pip install torch==2.10.0+cu128 torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### torchcodec warning

```
WARNING: torchcodec is not installed correctly
```

**Solution**: This is expected. WhisperTurbo uses in-memory audio loading to avoid torchcodec dependency. The warning is harmless.

### SSL certificate error

```
SSL certificate verification failed
```

**Solution**: The code patches SSL verification for HuggingFace downloads. If issues persist, set:
```bash
set HF_HUB_DISABLE_SSL_WARNINGS=1
```

### Diarization not working

```
Access to model pyannote/speaker-diarization-community-1 is restricted
```

**Solution**:
1. Visit [https://huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
2. Accept user conditions
3. Create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Set `HF_TOKEN` environment variable

### No audio captured

```
RMS: 0.000010 (silent)
```

**Solution**:
- Check that the correct audio device is selected
- For system audio: set Windows output to "CABLE Input"
- For microphone: check Windows sound settings
- Test with `python realtime_capture.py --list-devices`

### Translation returns Korean instead of English

This is a known bug in faster-whisper's `task="translate"`. The code uses a workaround (`language="en"` with `initial_prompt`) which generally works but may occasionally fail on certain audio.

## Performance

| Component | Speed | Model |
|-----------|-------|-------|
| Transcription | 30x realtime | large-v3-turbo (CTranslate2 int8_float16) |
| Translation | 31x realtime | large-v3-turbo (language="en" workaround) |
| Diarization | 2.5x realtime | pyannote/speaker-diarization-community-1 |
| End-to-end | ~2.5x realtime | Combined pipeline |

## License

MIT License - see LICENSE file.

## Contributing

See `plan.md` for the project roadmap and current phase.
