# WhisperTurbo Configuration

## Configuration File

Main configuration is located in `src/config.py`.

## Configuration Parameters

### Audio Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SAMPLE_RATE` | int | 16000 | Audio sample rate in Hz |
| `CHANNELS` | int | 1 | Number of audio channels (1 = mono) |
| `CHUNK_SIZE` | int | 1024 | Audio chunk size for processing |

### Whisper ASR Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `WHISPER_MODEL` | str | `deepdml/faster-whisper-large-v3-turbo-ct2` | Model identifier |
| `WHISPER_DEVICE` | str | `cuda` | Device to use (cuda/cpu) |
| `WHISPER_COMPUTE_TYPE` | str | `float16` | Compute type (float16/int8) |
| `WHISPER_TASK` | str | `translate` | Task type (transcribe/translate) |
| `WHISPER_LANGUAGE` | str | `ko` | Source language code |

### Diarization Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `DIARIZATION_MODEL` | str | `pyannote/speaker-diarization-community-1` | Model identifier |
| `DIARIZATION_WINDOW` | float | 15.0 | Analysis window in seconds |
| `DIARIZATION_OVERLAP` | float | 5.0 | Window overlap in seconds |

### GUI Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `GUI_PORT` | int | 5006 | Panel server port |
| `GUI_REFRESH_MS` | int | 100 | GUI refresh interval |
| `MAX_TABLE_ROWS` | int | 1000 | Maximum rows in table |

### Logging Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `LOG_LEVEL` | str | `INFO` | Logging level |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace token for diarization model access |
| `CUDA_VISIBLE_DEVICES` | No | GPU device selection |

## Example Configuration

```python
# src/config.py
SAMPLE_RATE = 16000
WHISPER_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
GUI_PORT = 5006
```

## Command-Line Options

Run with `--help` to see all options:

```bash
python main.py --help

Options:
  --no-gui           Disable GUI (headless mode)
  --no-diarization   Disable speaker diarization
  --gui-port PORT    GUI server port (default: 5006)
  --log-level LEVEL  Logging level (DEBUG, INFO, WARNING, ERROR)
```
