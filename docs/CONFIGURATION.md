# WhisperTurbo Configuration

## Configuration File

Main configuration is located in `src/config.py`.

## Configuration Parameters

### Audio Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SAMPLE_RATE` | int | 16000 | Audio sample rate in Hz |
| `CHANNELS` | int | 1 | Number of audio channels (1 = mono) |
| `DTYPE` | str | `float32` | Audio data type |
| `CHUNK_DURATION` | float | 0.1 | Audio chunk duration for input (seconds) |
| `BUFFER_DURATION` | float | 30.0 | Audio ring-buffer duration (seconds) |

### Whisper ASR Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `WHISPER_MODEL` | str | `deepdml/faster-whisper-large-v3-turbo-ct2` | Model identifier |
| `WHISPER_DEVICE` | str | `cuda` | Device to use (cuda/cpu) |
| `WHISPER_COMPUTE_TYPE` | str | `float16` | Compute type (float16/int8) |
| `WHISPER_TASK` | str | `translate` | Task type (transcribe/translate) |
| `WHISPER_LANGUAGE` | str | `ko` | Source language code |
| `WHISPER_BEAM_SIZE` | int | 5 | Beam size for decoding |
| `WHISPER_NO_SPEECH_THRESHOLD` | float | 0.6 | Probability threshold for no-speech detection |
| `WHISPER_LOGPROB_THRESHOLD` | float | -1.0 | Log-probability threshold for decoding |
| `WHISPER_COMPRESSION_RATIO_THRESHOLD` | float | 2.4 | Compression ratio threshold for decoding |
| `WHISPER_INITIAL_PROMPT` | str | `""` | Initial prompt for context (empty = none) |
| `ENABLE_CONTEXT_CARRY` | bool | `True` | Whether to carry context across segments |
| `CONTEXT_MAX_LENGTH` | int | 500 | Max context length (characters) |
| `CONTEXT_SEGMENT_COUNT` | int | 3 | Number of previous segments to carry |
| `MIN_PROCESSING_INTERVAL` | float | 2.0 | Minimum seconds between processing cycles |

### Diarization Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `DIARIZATION_MODEL` | str | `pyannote/speaker-diarization-community-1` | Model identifier |
| `DIARIZATION_WINDOW_SIZE` | float | 15.0 | Analysis window in seconds |
| `DIARIZATION_OVERLAP` | float | 5.0 | Window overlap in seconds |

### VAD Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `VAD_THRESHOLD` | float | 0.5 | VAD sensitivity threshold |
| `VAD_MIN_SPEECH_DURATION` | float | 0.1 | Min speech duration (seconds) |
| `VAD_MIN_SILENCE_DURATION` | float | 0.5 | Min silence duration (seconds) |

### GUI Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `GUI_REFRESH_RATE` | int | 100 | GUI refresh interval (ms) |
| `GUI_MAX_ROWS` | int | 100 | Maximum rows in the display table |
| `SPEAKER_COLORS` | tuple | (hex colors) | List of colors for speakers |

### Export Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `EXPORT_FORMATS` | tuple | `("csv", "jsonl", "srt")` | Supported export formats |

### Logging Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `LOG_LEVEL` | str | `INFO` | Logging level |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes* | HuggingFace token for diarization model access |
| `HUGGINGFACE_TOKEN` | No | Alternative token (deprecated) |
| `CUDA_VISIBLE_DEVICES` | No | GPU device selection |

*Required only if diarization is enabled.

## Example Configuration

```python
# src/config.py
SAMPLE_RATE = 16000
WHISPER_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
ENABLE_CONTEXT_CARRY = True
CONTEXT_MAX_LENGTH = 500
```

## Command-Line Options

### main.py Options

Run with `--help` to see all options:

```bash
python main.py --help

Options:
  --no-gui           Disable GUI (headless mode)
  --no-diarization   Disable speaker diarization
  --gui-port PORT    GUI server port (default: 5006)
  --benchmark        Run in benchmark mode (more verbose metrics)
  --log-level LEVEL  Logging level (DEBUG, INFO, WARNING, ERROR)
```

### launcher.py Options

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

## Export Formats

The system supports three export formats:

### CSV
Comma-separated values with metadata support. Contains columns: start, end, speaker, text, language, confidence, timestamp.

### JSONL
JSON Lines format with one JSON object per line. Easy to parse and process programmatically.

### SRT
SubRip subtitle format with timestamps and optional speaker labels. Compatible with video players.

To export from the GUI, use the export buttons and select your preferred format.
