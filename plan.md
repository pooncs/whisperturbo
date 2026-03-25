# WhisperTurbo Project Plan

## Ultimate Goal
A production-level, real-time, highly accurate transcriber, translator, and diarizer that is functional and user-friendly. Capable of handling face-to-face meetings as well as online meetings (via audio routing). Easy installation and usage.

---

## Phase 1: Core Pipeline Fix (✅ COMPLETE)
**Goal**: Fix diarization, translation, transcription for near real-time with high accuracy

### Tasks
- [x] Fix faster-whisper translate bug (task="translate" returns Korean instead of English)
- [x] Implement language="en" workaround with initial_prompt for proper translation
- [x] Use large-v3-turbo with CTranslate2 for speed (30x realtime transcription)
- [x] Integrate pyannote/speaker-diarization-community-1 for speaker diarization (2.5x realtime)
- [x] Use in-memory audio dict to avoid torchcodec dependency
- [x] Validate throughput: Transcribe 30x RT, Translate 31x RT, Diarize 2.5x RT
- [x] Fix VAD timestamp bug (samples vs milliseconds)
- [x] Fix generator exhaustion bug (segments_list vs segments)
- [x] Fix model compatibility (use HF model instead of cached CTranslate2)
- [x] Update config to use large-v3-turbo model

### Tests
- [x] Record 60s of Korean audio via VB-Audio Cable
- [x] Verify Korean transcription accuracy
- [x] Verify English translation produces English text
- [x] Verify speaker diarization detects 2+ speakers
- [x] Verify throughput meets near-realtime targets
- [x] Verify VAD finds speech segments correctly
- [x] Verify transcription returns non-empty segments

### Validation Checks
- [x] Audio capture RMS > 0.01 (not silent)
- [x] Korean segments detected > 0
- [x] English translation has no Korean characters
- [x] Diarization segments count > 0
- [x] Total processing time < 2x realtime for 60s audio

---

## Phase 2: Real-Time Streaming (⏳ IN PROGRESS)
**Goal**: Continuous audio streaming with live transcript output

### Tasks
- [x] Create realtime_capture.py with VB-Audio Cable loopback support
- [x] Implement 10-minute continuous capture
- [x] Write continuously to markdown file with timestamps
- [x] Show real-time throughput metrics (every 10s)
- [ ] Implement speaker labeling in output
- [ ] Add overlap processing to avoid missing words at chunk boundaries

### Tests
- [ ] Run 10-minute capture with Korean movie playing
- [ ] Verify continuous output file grows during capture
- [ ] Verify timestamps are sequential and accurate
- [ ] Verify no audio data loss (all segments captured)
- [ ] Verify throughput stays near-realtime throughout

### Validation Checks
- [ ] Output file exists and contains segments
- [ ] File size grows during capture
- [ ] No gaps > 5s between segments
- [ ] Average latency < 3s from audio to output

---

## Phase 3: Gradio UI Polish (⏳ IN PROGRESS)
**Goal**: Production-level user interface

### Tasks
- [x] Show all audio input devices (not just filtered)
- [x] Add device selection dropdown
- [x] Add real-time transcription display
- [x] Add speaker diarization labels
- [x] Add audio level indicator
- [x] Add export functionality for transcripts
- [x] Add file upload for batch processing
- [ ] Add live auto-refresh (every 0.5s)
- [ ] Add progress indicator during capture

### Tests
- [ ] Select each device and verify audio capture works
- [ ] Start translation and verify live output appears
- [ ] Stop translation and verify clean shutdown
- [ ] Export transcript and verify file format
- [ ] Upload audio file and verify translation

### Validation Checks
- [ ] UI loads without errors
- [ ] Device dropdown shows available devices
- [ ] Start button begins audio capture
- [ ] Live transcription appears within 5s
- [ ] Stop button cleanly stops all processing
- [ ] Export produces valid file

---

## Phase 4: Code Cleanup & Documentation (⏳ IN PROGRESS)
**Goal**: Clean codebase with comprehensive documentation

### Tasks
- [x] Move unused test files to old/ directory
- [x] Remove debug prints and unused imports
- [x] Standardize logging configuration
- [x] Write README.md with installation, usage, troubleshooting
- [x] Write plan.md with phases, todos, tests, validation

### Tests
- [ ] Run linting (ruff) and fix issues
- [ ] Verify all imports are used
- [ ] Verify README instructions work on clean install

### Validation Checks
- [ ] No files in root except core files
- [ ] No debug prints in production code
- [ ] README covers installation, usage, troubleshooting
- [ ] All .md files are complete and accurate

---

## Phase 5: Production Readiness (📋 PLANNED)
**Goal**: Easy installation, robust error handling, comprehensive testing

### Tasks
- [ ] Add proper error handling and logging throughout
- [ ] Add configuration file support (YAML/JSON)
- [ ] Add comprehensive unit tests
- [ ] Add integration tests
- [ ] Create one-click installer
- [ ] Add system tray support
- [ ] Add hotkey support (Ctrl+Shift+T to toggle)
- [ ] Add memory usage monitoring
- [ ] Add GPU memory monitoring
- [ ] Support multiple languages (not just Korean)

### Tests
- [ ] Unit tests for each component (>80% coverage)
- [ ] Integration test: full pipeline from audio to output
- [ ] Stress test: 60-minute continuous capture
- [ ] Error recovery test: simulate failures and verify graceful handling

### Validation Checks
- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] Stress test completes without memory leaks
- [ ] Error recovery test passes
- [ ] Performance meets targets: <2s latency, >20x realtime

---

## Phase 6: Advanced Features (📋 PLANNED)
**Goal**: Meeting support, multi-language, cloud integration

### Tasks
- [ ] Face-to-face meeting support (dual microphone input)
- [ ] Online meeting support (system audio routing)
- [ ] Multi-language detection and translation
- [ ] Real-time subtitle overlay
- [ ] Cloud storage integration (save transcripts to cloud)
- [ ] Meeting summary generation
- [ ] Action item extraction
- [ ] Speaker voiceprint registration

### Tests
- [ ] Face-to-face meeting with 2 speakers
- [ ] Online meeting via Zoom/Teams
- [ ] Multi-language detection (Korean/English/Japanese)
- [ ] Cloud sync verification

### Validation Checks
- [ ] Meeting transcripts are accurate
- [ ] Speaker identification is correct (>80%)
- [ ] Multi-language detection works
- [ ] Cloud sync is reliable

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Transcription RTF | >20x realtime | 30x ✅ |
| Translation RTF | >15x realtime | 31x ✅ |
| Diarization RTF | >2x realtime | 2.5x ✅ |
| End-to-end Latency | <3s | ~2.5s ✅ |
| Translation Accuracy | >90% | ~70% ⚠️ |
| Speaker Accuracy | >80% | TBD |
| Uptime (24h) | >99% | TBD |

---

## Known Issues

1. **Translation Quality**: The language="en" workaround produces English but accuracy varies. Need better prompting or model.

2. **Diarization Speed**: pyannote is 2.5x realtime which is slow. Can optimize with smaller windows.

3. **Audio Device Detection**: Some devices not detected properly. Need better device enumeration.

4. **Memory Usage**: Long sessions may accumulate memory. Need periodic cleanup.

---

## Environment

- **OS**: Windows 11
- **Python**: 3.11+ (conda env: whisper)
- **PyTorch**: 2.10.0+cu128
- **Whisper**: large-v3-turbo (CTranslate2 int8_float16)
- **Diarization**: pyannote/speaker-diarization-community-1
- **Audio**: VB-Audio Virtual Cable (loopback), Realtek Microphone Array (mic)

---

## Repository Structure

```
whisperturbo/
├── README.md              # Installation & usage guide
├── plan.md                # This file - project roadmap
├── main.py                # Pipeline entry point
├── launcher.py            # Environment validation & launch
├── gradio_gui.py          # Gradio UI
├── realtime_capture.py    # Real-time streaming capture
├── capture_cable.py       # VB-Audio Cable capture (test)
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration
│   ├── audio_input.py     # Audio input handler
│   ├── whisper_asr.py     # Whisper ASR wrapper
│   ├── diarization.py     # Speaker diarization
│   ├── fusion.py          # ASR + diarization fusion
│   └── postprocess.py     # Post-processing
├── old/                   # Deprecated files
└── tests/                 # Test files
```
