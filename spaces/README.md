---
title: WhisperTurbo
emoji: 🎙️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
tags:
- speech-recognition
- transcription
- faster-whisper
- audio
- nlp
model_class: WhisperTurbo
---

# WhisperTurbo - Speech Transcription

Real-time speech transcription using Faster-Whisper with browser microphone support.

## Features
- Browser-based microphone recording
- Multi-language support (Auto-detect, Korean, Japanese, Chinese, English, etc.)
- Multiple model sizes (tiny, base, small, medium, large-v3)
- Voice Activity Detection (VAD) filtering

## Usage
1. Open the Space URL
2. Select your preferred model size (default: base)
3. Choose source language or use auto-detect
4. Click the microphone button to record
5. Click "Transcribe" to process the audio

## Model Sizes
- **tiny**: Fastest, lowest accuracy
- **base**: Good balance of speed and accuracy (recommended)
- **small**: Better accuracy, slower
- **medium**: High accuracy, much slower
- **large-v3**: Best accuracy, requires more resources

## Requirements
- Microphone access in browser
- Internet connection for model download