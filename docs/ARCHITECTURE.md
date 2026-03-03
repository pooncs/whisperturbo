# WhisperTurbo Architecture

## Overview

WhisperTurbo is a real-time Korean → English speech translation system that combines automatic speech recognition (ASR), speaker diarization, and machine translation in a single pipeline.

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐
│  Microphone    │────>│   AudioInput     │
│  (sounddevice) │     │   (Ring Buffer)  │
└─────────────────┘     └────────┬─────────┘
                                 │
               ┌──────────────────┼──────────────────┐
               │                  │                  │
               v                  v                  v
     ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐
     │     VAD         │  │  Whisper-Turbo  │  │  Diarization│
     │ (Silero VAD)    │  │ (ASR+Translate) │  │(pyannote)   │
     └────────┬────────┘  └────────┬─────────┘  └──────┬──────┘
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  │
                                  v
                         ┌─────────────────┐
                         │     Fusion      │
                         │ (Speaker Map)   │
                         └────────┬────────┘
                                  │
                                  v
                         ┌─────────────────┐
                         │   Panel GUI     │
                         │  (Real-time)    │
                         └─────────────────┘
```

## Components

### 1. AudioInput (`src/audio_input.py`)

Handles real-time audio capture from the microphone using `sounddevice`.

- **Sample Rate**: 16kHz (required for Whisper)
- **Format**: Mono Float32
- **Buffer**: Ring buffer for continuous streaming
- **VAD Integration**: Compatible with Faster-Whisper's Silero VAD

### 2. Whisper ASR (`src/whisper_asr.py`)

Faster-Whisper implementation for speech recognition and translation.

- **Model**: `deepdml/faster-whisper-large-v3-turbo-ct2`
- **Task**: `translate` (Korean → English)
- **Device**: CUDA
- **Compute Type**: float16

**Context Carry Mechanism**:
The ASR module maintains context across segments for improved translation continuity:
- Stores previous transcription text in `_previous_text` list
- Builds initial prompt from recent segments
- Configurable via `ENABLE_CONTEXT_CARRY`, `CONTEXT_MAX_LENGTH`, and `CONTEXT_SEGMENT_COUNT`
- Passed to Whisper as `initial_prompt` parameter

### 3. Diarization (`src/diarization.py`)

Speaker diarization using pyannote.audio.

- **Model**: `pyannote/speaker-diarization-community-1`
- **Window Size**: 15 seconds
- **Overlap**: 5 seconds
- **Token**: Requires HuggingFace token (HF_TOKEN)
- **Async Processing**: Runs asynchronously to avoid blocking ASR

### 4. Fusion (`src/fusion.py`)

Merges ASR transcripts with speaker labels using dominant-overlap mapping.

- Each ASR segment `[t0, t1]` is assigned the speaker with the longest overlap
- Produces unified log entries: `[time] [speaker] translated_text`
- Applies post-processing via `postprocess.py`

**Deduplication**:
The fusion module prevents duplicate segment emission:
- Tracks `_last_emitted_end_time` to avoid re-emitting processed segments
- Uses epsilon threshold (0.1s) for floating-point comparison
- Filters segments with `asr_seg.end <= _last_emitted_end_time - epsilon`

### 5. Post-processing (`src/postprocess.py`)

Text post-processing utilities for cleaning transcription output.

- **normalize_whitespace()**: Collapses multiple whitespace characters to single space
- **trim_repetitions()**: Removes consecutive duplicate words
- **merge_short_segments()**: Merges short segments from same speaker with small time gaps

### 6. GUI (`src/gui.py`)

Real-time visualization using Panel.

- **Display**: Scrolling table with timestamp, speaker, translated text
- **KPIs**: Latency, RTF (Real-Time Factor), segment count, active speakers
- **Export**: CSV, JSONL, SRT formats

### 7. Launcher (`launcher.py`)

Environment validation and user-friendly startup.

- Python version validation (3.9+)
- CUDA availability checking
- Audio device enumeration
- Dependency verification
- Model presence checking
- Automatic browser opening
- Colored console output

## Data Flow

1. **Audio Capture**: Microphone input → Ring buffer
2. **VAD Detection**: Silero VAD identifies speech segments
3. **ASR Processing**: Whisper processes segments for translation
4. **Context Carry**: Previous segments feed into initial prompt
5. **Diarization**: pyannote analyzes audio windows for speakers
6. **Fusion**: Merge translations with speaker labels
7. **Post-processing**: Clean text (whitespace, repetitions)
8. **Display**: Update GUI with new entries

## Threading Model

- **Main Thread**: GUI event loop / Launcher control
- **Audio Thread**: Continuous audio capture (sounddevice callback)
- **ASR Thread**: Async transcription processing
- **Diarization Thread**: Async speaker analysis

## Benchmark Mode

The `--benchmark` flag enables verbose performance logging:

- Logs latency, processing time, audio duration, and RTF every cycle
- Prints aggregate statistics every 5 cycles
- Useful for performance tuning and profiling

Example output:
```
METRICS: latency=0.45s processing=0.32s audio_duration=5.00s rtf=0.06 segments=2
BENCHMARK: Avg Latency: 0.48s, Avg RTF: 0.07x, Total Segments: 25
```

## Performance Characteristics

- **Typical Latency**: < 1 second
- **Diarization Coverage**: ≥ 90%
- **Real-Time Factor (RTF)**: < 1.0 (faster than real-time)
- **Context Window**: Configurable (default: last 3 segments, max 500 chars)
