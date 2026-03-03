# Real-Time Speech Translation: Windows Deployment Plan
**Target:** Windows Laptop (NVIDIA RTX 4090)
**Goal:** Low-latency Korean → English translation using Whisper-Turbo (Faster-Whisper), with real-time GUI logging and speaker diarization.
**Architecture:** Single-Pipeline (Streaming ASR+Translate + Diarization + GUI)
---
## 1. Executive Summary & Recommendation
This plan uses **Whisper-large-v3-turbo** via **Faster-Whisper (CTranslate2)** as a single low-latency ASR+translation engine. Diarization is powered by **pyannote.audio**, and a Panel GUI provides real-time display of translated, speaker-attributed logs.
---
## 2. Architecture Diagram
```mermaid
graph TD
 Mic[Microphone Input] --> VAD[Silero VAD (Faster-Whisper)]
 VAD --> FW[Whisper-Turbo (Translate Mode)]
 FW --> Segs[English Segments w/ timestamps]
 Segs --> GUI[Panel GUI Live Log]
 FW --> AudioWin[Audio Window Buffer]
 AudioWin --> DIA[pyannote.audio Diarization]
 DIA --> SpeakerTags[Speaker Segments]
 SpeakerTags --> Merge[Merge Speakers + ASR]
 Merge --> GUI
```
---
## 3. Phased Implementation Plan

### Phase 1: Audio & VAD Foundation
- Create `AudioInput` class using `sounddevice` (16kHz Mono Float32).
- Integrate Faster-Whisper’s built-in Silero VAD via `VadOptions`.
- Maintain a rolling buffer for feeding diarization windows.

### Phase 2: Whisper-Turbo (Faster-Whisper) ASR+Translate
- Load `deepdml/faster-whisper-large-v3-turbo-ct2`.
- Use `task="translate"` for direct Korean→English.
- Emit English text + timestamps.

### Phase 3: Speaker Diarization (pyannote.audio)
- Load `pyannote/speaker-diarization-community-1`.
- Run on rolling windows (e.g., 15s) asynchronously.
- Match diarization segments to ASR segments via timestamp overlap.

### Phase 4: Fusion (Speaker + Translation)
- Dominant-overlap mapping:
  - Each ASR segment `[t0,t1]` gets the speaker with the longest overlap.
- Produce unified log entries:
  - `[time] [speaker] translated_text`.

### Phase 5: Real-Time GUI (Panel)
- GUI Elements:
  - Live scrolling table showing time, speaker, translated text.
  - KPIs: latency, RTF.
- Use Panel periodic callbacks to update display.
- Allow export to CSV/JSONL/SRT.

### Phase 6: Deployment (Windows)
- Package using PyInstaller.
- Bundle CUDA/CTranslate2/ONNX runtime dependencies.
- Bundle or auto-download models.
---
## 4. Implementation Prompts
### Prompt A: AudioInput
```
Create AudioInput using sounddevice:
1. 16kHz Float32.
2. Callback captures audio into ring buffer.
3. Compatible with Faster-Whisper VAD.
```
### Prompt B: Whisper-Turbo ASR
```
Use Faster-Whisper:
model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda", compute_type="float16")
Use task="translate".
```
### Prompt C: Diarization
```
Load pyannote pipeline:
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=HF_TOKEN)
```
---
## 5. Validation & Metrics
- Latency < 1s typical.
- Diarization coverage ≥ 90%.
- Exported logs match timestamps + speaker labels.
---
## 6. Deployment Checklist
- Ensure CUDA runtime DLLs present.
- Ensure HF token prompt for first-run diarization.
- Provide GUI launcher script.
