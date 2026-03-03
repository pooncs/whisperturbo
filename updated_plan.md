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

This repository already contains a working end-to-end pipeline:
- Audio: [audio_input.py](file:///c:/Users/hmgics/projects/whisperturbo/src/audio_input.py)
- ASR/Translate: [whisper_asr.py](file:///c:/Users/hmgics/projects/whisperturbo/src/whisper_asr.py)
- Diarization: [diarization.py](file:///c:/Users/hmgics/projects/whisperturbo/src/diarization.py)
- Fusion: [fusion.py](file:///c:/Users/hmgics/projects/whisperturbo/src/fusion.py)
- GUI: [gui.py](file:///c:/Users/hmgics/projects/whisperturbo/src/gui.py)
- Orchestration: [main.py](file:///c:/Users/hmgics/projects/whisperturbo/main.py)

The phased plan below focuses on improving:
1) accuracy of translation, 2) speed of translation, 3) deployment readiness, 4) GUI, 5) unit tests, 6) code cleanliness.

### Phase 0: Baseline, Metrics, and Guardrails (do first)
- [x] Establish current unit-test baseline.
- [ ] Add a “golden metrics” script or mode that prints latency/RTF consistently.
- [ ] Define and enforce a single timebase for all timestamps (audio-time seconds).

**Where to improve**
- Pipeline has hard-coded loop timings (process interval / window sizes) in [main.py](file:///c:/Users/hmgics/projects/whisperturbo/main.py#L130-L210).

**Evidence of completion**
- [x] `python -m pytest -q` passes (current baseline: 112 passed).
- [ ] `python main.py --no-gui --no-diarization --log-level DEBUG` prints per-segment latency + RTF lines.

### Phase 1: Timestamp Correctness + Duplicate Suppression (accuracy + diarization correctness)
- [ ] Fix ASR segment timestamps to be globally aligned to the audio ring-buffer timebase.
- [ ] Prevent re-processing the same audio repeatedly (current loop transcribes overlapping “recent audio” windows).
- [ ] Make fusion operate on aligned timestamps (ASR and diarization must overlap in the same coordinate system).

**Where to improve**
- ASR segments returned by Faster-Whisper are relative to the provided audio chunk, but diarization segments are offset by `diarization_timestamp` in [main.py](file:///c:/Users/hmgics/projects/whisperturbo/main.py#L172-L190). This mismatch undermines speaker mapping in [fusion.py](file:///c:/Users/hmgics/projects/whisperturbo/src/fusion.py#L30-L103).

**Evidence of completion**
- [ ] New unit test proves fusion chooses the correct speaker when ASR timestamps are offset.
- [ ] New unit test proves dedupe logic prevents emitting the same text twice across overlapping windows.

### Phase 2: Translation Accuracy Improvements (model params + post-processing)
- [ ] Surface decoding and hallucination-guard parameters in config (beam size, no-speech threshold, logprob threshold, compression ratio threshold).
- [ ] Add an optional “context carry” mechanism across segments (prompting/conditioning) to reduce fragmented translations.
- [ ] Add lightweight post-processing: whitespace normalization, repetition trimming, and segment merge rules.

**Where to improve**
- ASR currently uses fixed `beam_size=5` and no guard thresholds in [whisper_asr.py](file:///c:/Users/hmgics/projects/whisperturbo/src/whisper_asr.py#L78-L130).

**Evidence of completion**
- [ ] Unit tests cover post-processing (repetition trimming, merge behavior).
- [ ] A small deterministic “fixture audio” test validates stable translation output shape (mocked model output is acceptable).

### Phase 3: Speed Improvements (async diarization + smaller work units)
- [ ] Move diarization off the critical path (async or lower-cadence rolling windows).
- [ ] Only run ASR on speech regions (VAD-driven chunking) rather than fixed 5s windows.
- [ ] Introduce backpressure/queueing to avoid piling work when the GPU is busy.

**Where to improve**
- Diarization can block the pipeline because it runs inline per cycle in [main.py](file:///c:/Users/hmgics/projects/whisperturbo/main.py#L172-L190), despite having async primitives available in [diarization.py](file:///c:/Users/hmgics/projects/whisperturbo/src/diarization.py#L129-L156).

**Evidence of completion**
- [ ] A benchmark mode prints p50/p95 latency and RTF after N segments.
- [ ] Optional config toggles: “fast mode” (lower beam / compute_type variants) vs “accuracy mode”.

### Phase 4: GUI Enhancements (reliability + operator UX)
- [ ] Ensure the table reliably refreshes (periodic callback wired to `_refresh_table`).
- [ ] Add operator controls: pause/resume, clear, diarization toggle indicator, and error/status area.
- [ ] Add speaker UX: consistent colors per speaker, filters, and “active speaker” list.
- [ ] Make exports explicit: choose output directory and include run metadata (model names, settings, start time).

**Where to improve**
- GUI defines `_refresh_table` but does not schedule it; `GUI_REFRESH_RATE` is also unused in [gui.py](file:///c:/Users/hmgics/projects/whisperturbo/src/gui.py#L174-L206).

**Evidence of completion**
- [ ] Unit test verifies refresh callback is registered (mock Panel state).
- [ ] Manual check: GUI updates without clicking and exports write files successfully.

### Phase 5: Deployment Readiness (Windows, offline-friendly, reproducible)
- [ ] Make the PyInstaller build reproducible in-repo (spec file generation or checked-in spec).
- [ ] Align build scripts with repo contents (`build.py`/`build.bat` currently reference `whisperturbo.spec` which is absent).
- [ ] Align “model prefetch” tooling with `requirements.txt` (download script uses packages not declared).
- [ ] Add first-run checks: CUDA availability, audio device selection, HF token validation, and graceful fallback to `--no-diarization`.

**Where to improve**
- `whisperturbo.spec` is referenced by [build.py](file:///c:/Users/hmgics/projects/whisperturbo/build.py) and [build.bat](file:///c:/Users/hmgics/projects/whisperturbo/build.bat) but is not present in this repo checkout.
- `download_models.py` relies on `huggingface_hub` and `tqdm`, but they are not in [requirements.txt](file:///c:/Users/hmgics/projects/whisperturbo/requirements.txt).
- `launcher.py` has a banner bug (uses undefined color variables) in [launcher.py](file:///c:/Users/hmgics/projects/whisperturbo/launcher.py#L30-L42).

**Evidence of completion**
- [ ] `python build.py` produces a runnable `dist\\WhisperTurbo\\WhisperTurbo.exe`.
- [ ] A clean machine runbook: install Python, install deps, prefetch models, run launcher.

### Phase 6: Test Coverage + Code Cleanliness (keep velocity)
- [ ] Add integration-ish tests for the real pipeline invariants (timebase alignment, dedupe, diarization async behavior).
- [ ] Introduce lint/format/typecheck tooling (e.g., ruff/black/mypy) with a single command entrypoint.
- [ ] Remove dead or duplicate entrypoints, align docs vs code config names.

**Where to improve**
- Unit tests exist and pass, but key real-time invariants are not asserted (most pipeline tests are heavily mocked) in [test_translation_pipeline.py](file:///c:/Users/hmgics/projects/whisperturbo/tests/test_translation_pipeline.py).
- Docs drift: [CONFIGURATION.md](file:///c:/Users/hmgics/projects/whisperturbo/docs/CONFIGURATION.md) references settings that do not exist in [config.py](file:///c:/Users/hmgics/projects/whisperturbo/src/config.py).

**Evidence of completion**
- [ ] `python -m pytest -q` still passes.
- [ ] `python -m ruff check .` (or equivalent) passes.
- [ ] `python -m mypy src` (or equivalent) passes.
---
## 4. Implementation Prompts

### Prompt 0: Baseline + Metrics (agentic)
```text
Goal: establish baseline performance and a single timestamp convention.

1) Confirm tests pass locally:
   - python -m pytest -q

2) Add a CLI flag or log mode that prints per-cycle metrics:
   - latency_seconds
   - last_processing_time
   - audio_duration_seconds
   - rtf
   - segments_emitted

3) Add tests that validate the metrics function is deterministic given fixed inputs.

Files likely touched:
- main.py
- src/whisper_asr.py
- tests/

Definition of done:
- pytest passes
- logs show metrics every cycle in DEBUG mode
```

### Prompt 1: Timestamp Alignment + Dedupe (agentic)
```text
Goal: fix speaker mapping correctness and prevent duplicate translation emissions.

Implement:
- Track an audio-time cursor (seconds since AudioInput start).
- When extracting audio windows, compute the window's start_time (audio-time).
- Offset ASR segment start/end by window_start_time.
- Keep a “last_emitted_end_time” and drop segments that end <= cursor - epsilon.

Update Fusion inputs so ASR segments and diarization segments share the same timebase.

Add tests:
- A fusion test where ASR is offset and diarization overlaps only after offset.
- A pipeline test where overlapping windows do not duplicate segments.

Files likely touched:
- main.py
- src/fusion.py (only if needed)
- tests/test_fusion.py
- tests/test_translation_pipeline.py

Definition of done:
- new tests pass
- manual run shows stable speakers instead of frequent UNKNOWN/flip-flop
```

### Prompt 2: Accuracy Tuning (agentic)
```text
Goal: reduce hallucinations and improve translation quality without sacrificing latency.

Implement config-driven decode options:
- beam_size
- vad thresholds already exist; add no_speech/logprob/compression thresholds
- optional "condition_on_previous_text" / "initial_prompt" strategy

Add post-processing:
- trim whitespace
- collapse repeats
- merge short adjacent segments with same speaker

Add tests:
- post-processing is deterministic
- thresholds are forwarded into WhisperModel.transcribe (mock)

Files likely touched:
- src/config.py
- src/whisper_asr.py
- src/fusion.py (merge rule)
- tests/test_whisper_asr.py
- tests/test_fusion.py
```

### Prompt 3: Speed + Async Diarization (agentic)
```text
Goal: keep translation latency low by removing diarization from the critical path.

Implement:
- Use DiarizationHandler.process_async on a rolling window at a slower cadence.
- Cache the latest diarization segments and reuse them for fusion.
- Add backpressure: if diarization is busy, skip rather than block.

Add tests:
- diarization is invoked asynchronously (mock thread start)
- pipeline continues to emit translations when diarization is busy/fails

Files likely touched:
- src/diarization.py
- main.py
- tests/test_diarization.py
- tests/test_translation_pipeline.py
```

### Prompt 4: GUI Refresh + UX (agentic)
```text
Goal: ensure the GUI updates reliably and is operator-friendly.

Implement:
- Register a periodic callback to call _refresh_table at CONFIG.GUI_REFRESH_RATE.
- Add a speaker->color mapping and show it in the table.
- Add pause/resume and a status/error log pane.

Add tests:
- verify periodic callback registration (mock panel)
- verify export functions create files (already covered for Fusion exports)

Files likely touched:
- src/gui.py
- tests/ (new GUI-focused tests)
```

### Prompt 5: Windows Packaging (agentic)
```text
Goal: produce a runnable Windows executable with predictable behavior.

Implement:
- Make build scripts self-contained:
  - generate PyInstaller spec if missing OR create an equivalent onefile/onedir build command
  - ensure required DLLs/data files are included (torch, ctranslate2, sounddevice/portaudio)
- Align dependency lists:
  - ensure download_models.py dependencies exist in requirements or a separate dev/tool requirements file
- Fix launcher entrypoints:
  - launcher.py banner bug
  - launcher.bat missing dependency variable assignment bug

Add tests:
- lightweight tests for launcher argument parsing and environment checks (mocked)

Files likely touched:
- build.py / build.bat
- launcher.py / launcher.bat
- requirements.txt
- tests/

Definition of done:
- dist\\WhisperTurbo\\WhisperTurbo.exe runs and opens GUI, or prints clear error messages.
```
---
## 5. Validation & Metrics
- [ ] Latency: p50 < 1.0s and p95 < 2.0s on RTX 4090.
- [ ] RTF: p50 < 1.0 and p95 < 1.2 (configurable target).
- [ ] Speaker mapping: < 5% UNKNOWN for speech segments in a 2-minute sample.
- [ ] Export integrity: CSV/JSONL/SRT contain aligned timestamps and speaker labels.

**How to measure**
- `python main.py --log-level DEBUG` and record p50/p95 from emitted metrics.
- Optional: run with `--no-diarization` to isolate ASR performance.
---
## 6. Deployment Checklist
- [ ] `python -m pip install -r requirements.txt` completes on a clean Windows machine.
- [ ] `HF_TOKEN` is requested or validated only when diarization is enabled.
- [ ] GPU path works and CPU fallback is explicit and documented in logs.
- [ ] `python download_models.py --token <HF_TOKEN>` works or fails with actionable error (dependencies installed).
- [ ] `python build.py` produces `dist\\WhisperTurbo\\WhisperTurbo.exe`.
- [ ] Launcher scripts start the app and surface clear diagnostics.
