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

### 3. Diarization (`src/diarization.py`)

Speaker diarization using pyannote.audio.

- **Model**: `pyannote/speaker-diarization-community-1`
- **Window Size**: 15 seconds
- **Overlap**: 5 seconds
- **Token**: Requires HuggingFace token (HF_TOKEN)

### 4. Fusion (`src/fusion.py`)

Merges ASR transcripts with speaker labels using dominant-overlap mapping.

- Each ASR segment `[t0, t1]` is assigned the speaker with the longest overlap
- Produces unified log entries: `[time] [speaker] translated_text`

### 5. GUI (`src/gui.py`)

Real-time visualization using Panel.

- **Display**: Scrolling table with timestamp, speaker, translated text
- **KPIs**: Latency, RTF (Real-Time Factor), segment count, active speakers
- **Export**: CSV, JSONL, SRT formats

## Data Flow

1. **Audio Capture**: Microphone input → Ring buffer
2. **VAD Detection**: Silero VAD identifies speech segments
3. **ASR Processing**: Whisper processes segments for translation
4. **Diarization**: pyannote analyzes audio windows for speakers
5. **Fusion**: Merge translations with speaker labels
6. **Display**: Update GUI with new entries

## Threading Model

- **Main Thread**: GUI event loop
- **Audio Thread**: Continuous audio capture (sounddevice callback)
- **ASR Thread**: Async transcription processing
- **Diarization Thread**: Async speaker analysis

## Performance Characteristics

- **Typical Latency**: < 1 second
- **Diarization Coverage**: ≥ 90%
- **Real-Time Factor (RTF)**: < 1.0 (faster than real-time)
