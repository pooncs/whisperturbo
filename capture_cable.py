"""Proper translation using a two-step approach with diarization."""

import os
import time
import numpy as np
import soundfile as sf
from datetime import datetime
from faster_whisper import WhisperModel
import sounddevice as sd
from scipy import signal
import torch
from pyannote.audio import Pipeline

OUTPUT_DIR = r"C:\Users\hmgics\projects\test_outputs"
DURATION = 60  # 60 seconds

# VB-Audio Cable Output device (loopback input)
DEVICE = 1
DEVICE_NAME = "CABLE Output (VB-Audio Virtual Cable)"

print("=" * 60)
print("System Audio Capture via VB-Audio Cable Loopback")
print("=" * 60)
print(f"Device: {DEVICE_NAME}")
print(f"Recording {DURATION} seconds...")
print("Make sure Korean movie is playing!")

# Record at native sample rate
sample_rate = 44100
t0 = time.time()
audio = sd.rec(
    DURATION * sample_rate,
    samplerate=sample_rate,
    channels=2,
    device=DEVICE,
    dtype=np.float32,
)
sd.wait()
capture_time = time.time() - t0

# Convert stereo to mono
audio_mono = np.mean(audio, axis=1)

# Check levels
rms = np.sqrt(np.mean(audio_mono**2))
max_amp = np.max(np.abs(audio_mono))

print(f"Recording complete!")
print(f"  RMS: {rms:.6f}")
print(f"  Max: {max_amp:.6f}")

# Proper resampling using scipy (to preserve pitch)
target_rate = 16000
num_samples = int(len(audio_mono) * target_rate / sample_rate)
audio_16k = signal.resample(audio_mono, num_samples)

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
audio_path = os.path.join(OUTPUT_DIR, f"cable_{timestamp}.wav")
sf.write(audio_path, (audio_16k * 32767).astype(np.int16), target_rate)
print(f"Audio: {audio_path}")

# Load model - use large-v3-turbo with CTranslate2 for speed
print("Loading model...")
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")

# Step 1: Transcribe in Korean
print("Step 1: Transcribing in Korean...")
t0 = time.time()
segments_ko, info = model.transcribe(
    audio_path,
    language="ko",
    task="transcribe",  # Use 'transcribe' not 'translate'
    vad_filter=False,
)
segments_ko = list(segments_ko)
transcribe_time = time.time() - t0

print(f"Detected: {info.language}, Segments: {len(segments_ko)}")

# Save Korean transcription
korean_text = " ".join([s.text for s in segments_ko])

ko_path = os.path.join(OUTPUT_DIR, f"cable_{timestamp}_korean.txt")
with open(ko_path, "w", encoding="utf-8") as f:
    f.write(f"Language: {info.language}, Prob: {info.language_probability:.2f}\n\n")
    for s in segments_ko:
        f.write(f"[{s.start:.2f}s - {s.end:.2f}s] {s.text}\n")

print(f"Korean: {ko_path}")

# Step 2: Translate to English using translate task
print("Step 2: Translating to English...")
t0 = time.time()
segments_en, info = model.transcribe(
    audio_path,
    language="ko",
    task="translate",  # This should translate
    vad_filter=False,
)
segments_en = list(segments_en)
translate_time = time.time() - t0

print(f"Translation segments: {len(segments_en)}")

# Check if translation worked - if output is same as Korean, it's broken
translation = " ".join([s.text for s in segments_en])

# Check if the translation is actually in English (not Korean)
# If the translation looks like Korean, try a workaround
is_korean = any(
    "\uac00" <= c <= "\ud7a3" for c in translation[:50]
)  # Check for Korean characters

if is_korean:
    print("WARNING: Translation appears to be in Korean, trying workaround...")
    # Workaround: Use English as target language with initial prompt
    t0 = time.time()
    segments_en, info = model.transcribe(
        audio_path,
        language="en",  # Force English output
        task="transcribe",
        initial_prompt="Translate the following Korean speech to natural, fluent English. Maintain the original meaning, emotion, and speaking style.",
        beam_size=5,
        best_of=5,
        vad_filter=False,
    )
    segments_en = list(segments_en)
    translate_time = time.time() - t0  # Override with workaround time
    translation = " ".join([s.text for s in segments_en])
    print(f"Workaround segments: {len(segments_en)}")

# Set HF_TOKEN for pyannote diarization (same as launcher.py)
import os

DEFAULT_HF_TOKEN = ""
if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = DEFAULT_HF_TOKEN

# Step 3: Speaker Diarization using pyannote
# Use in-memory audio to avoid torchcodec issues
print("Step 3: Speaker Diarization...")
t0 = time.time()

# Load pyannote pipeline
from pyannote.audio import Pipeline

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1", token=True
)

# Pre-load audio in memory to avoid torchcodec dependency
waveform, sr = sf.read(audio_path)
# Convert stereo to mono if needed
if len(waveform.shape) > 1:
    waveform = np.mean(waveform, axis=1)
# Resample to 16k if needed
if sr != 16000:
    num_samples = int(len(waveform) * 16000 / sr)
    waveform = signal.resample(waveform, num_samples)
    sr = 16000

# pyannote expects (channels, samples) but we need to match expected format
audio_dict = {
    "waveform": torch.from_numpy(waveform).float().unsqueeze(0),
    "sample_rate": sr,
}

# Run diarization with in-memory audio
diarization = diarization_pipeline(audio_dict)
diarize_time = time.time() - t0

# Extract diarization segments
diarization_text = []
try:
    # DiarizeOutput has speaker_diarization as Annotation object
    annotation = diarization.speaker_diarization
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        diarization_text.append(f"[{turn.start:.2f}s - {turn.end:.2f}s] {speaker}")
except Exception as e:
    print(f"Diarization extraction error: {e}")
    diarization_text = [f"Diarization completed with {len(annotation)} tracks"]

diarization_str = "\n".join(diarization_text)
md_path = os.path.join(OUTPUT_DIR, f"cable_{timestamp}.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(f"# System Audio Translation (VB-Audio Cable)\n\n")
    f.write(f"**Device**: {DEVICE_NAME} (device {DEVICE})\n")
    f.write(f"**Date**: {datetime.now()}\n")
    f.write(f"**Duration**: {DURATION}s\n")
    f.write(f"**RMS**: {rms:.6f}\n")
    f.write(f"**Max**: {max_amp:.6f}\n\n")
    f.write(f"Audio: [cable_{timestamp}.wav](cable_{timestamp}.wav)\n\n")
    f.write(f"## Korean Transcription\n\n{korean_text}\n\n")
    f.write(f"## English Translation\n\n{translation}\n\n")
    f.write(f"## Speaker Diarization\n\n{diarization_str}\n")

# Save English segments separately
seg_path = os.path.join(OUTPUT_DIR, f"cable_{timestamp}_en_segments.txt")
with open(seg_path, "w", encoding="utf-8") as f:
    for s in segments_en:
        f.write(f"[{s.start:.2f}s - {s.end:.2f}s] {s.text}\n")

print(f"Result: {md_path}")

# Throughput reporting
print("\n" + "=" * 60)
print("THROUGHPUT METRICS")
print("=" * 60)
print(f"Audio Capture:    {capture_time:.2f}s ({DURATION/capture_time:.1f}x realtime)")
print(
    f"Transcribe (ko): {transcribe_time:.2f}s ({DURATION/transcribe_time:.1f}x realtime)"
)
print(
    f"Translate (en):  {translate_time:.2f}s ({DURATION/translate_time:.1f}x realtime)"
)
print(f"Diarization:      {diarize_time:.2f}s ({DURATION/diarize_time:.1f}x realtime)")
total_time = capture_time + transcribe_time + translate_time + diarize_time
print(f"Total Time:       {total_time:.2f}s ({DURATION/total_time:.1f}x realtime)")
print("=" * 60)

print("\n=== COMPLETE ===")
print(f"Audio: {audio_path}")
print(f"Translation: {md_path}")
