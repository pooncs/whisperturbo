import logging
from dataclasses import dataclass, field
from typing import Optional

from .diarization import SpeakerSegment
from .postprocess import merge_short_segments, normalize_whitespace, trim_repetitions
from .whisper_asr import TranscriptionSegment

logger = logging.getLogger(__name__)


@dataclass
class TranslatedSegment:
    start: float
    end: float
    source_text: str = ""  # Original transcription (e.g., Korean)
    target_text: str = ""  # Translated text (e.g., English)
    source_language: str = ""
    target_language: str = "en"
    speaker: Optional[str] = None
    confidence: float = 1.0
    timestamp: float = 0.0
    correction_status: str = "fast"  # 'fast' or 'corrected'


class Fusion:
    def __init__(self, min_overlap: float = 0.3, epsilon: float = 0.1):
        self.min_overlap = min_overlap
        self.epsilon = epsilon
        self._pending_segments: list[TranslatedSegment] = []
        self._speaker_history: dict[str, float] = {}
        self._last_emitted_end_time: float = 0.0

    def _calculate_overlap(
        self, asr_seg: TranscriptionSegment, speaker_seg: SpeakerSegment
    ) -> float:
        overlap_start = max(asr_seg.start, speaker_seg.start)
        overlap_end = min(asr_seg.end, speaker_seg.end)

        if overlap_end <= overlap_start:
            return 0.0

        overlap_duration = overlap_end - overlap_start
        asr_duration = asr_seg.end - asr_seg.start

        return overlap_duration / asr_duration if asr_duration > 0 else 0.0

    def _get_dominant_speaker(
        self,
        asr_segment: TranscriptionSegment,
        speaker_segments: list[SpeakerSegment],
    ) -> Optional[str]:
        if not speaker_segments:
            return None

        best_speaker = None
        best_overlap = 0.0

        for speaker_seg in speaker_segments:
            overlap = self._calculate_overlap(asr_segment, speaker_seg)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker_seg.speaker

        if best_overlap < self.min_overlap:
            return None

        if best_speaker:
            self._speaker_history[best_speaker] = asr_segment.end

        return best_speaker

    def _get_last_speaker(self) -> Optional[str]:
        if not self._speaker_history:
            return None

        return max(self._speaker_history.items(), key=lambda x: x[1])[0]

    def fuse(
        self,
        asr_segments: list[TranscriptionSegment],
        translated_segments: list[TranscriptionSegment],
        speaker_segments: list[SpeakerSegment],
        timestamp: float,
        source_language: str = "",
        target_language: str = "en",
    ) -> list[TranslatedSegment]:
        fused: list[TranslatedSegment] = []

        # Build translation lookup by time
        translation_lookup = {}
        for trans_seg in translated_segments:
            key = (round(trans_seg.start, 1), round(trans_seg.end, 1))
            translation_lookup[key] = trans_seg.text.strip()

        for asr_seg in asr_segments:
            if not asr_seg.text:
                continue

            if asr_seg.end <= self._last_emitted_end_time - self.epsilon:
                continue

            # Find matching translation
            key = (round(asr_seg.start, 1), round(asr_seg.end, 1))
            target_text = translation_lookup.get(key, "")

            # If no exact match, find closest translation
            if not target_text:
                for trans_seg in translated_segments:
                    if (abs(trans_seg.start - asr_seg.start) < 1.0 and
                        abs(trans_seg.end - asr_seg.end) < 1.0):
                        target_text = trans_seg.text.strip()
                        break

            speaker = self._get_dominant_speaker(asr_seg, speaker_segments)

            if speaker is None:
                speaker = self._get_last_speaker()

            fused.append(
                TranslatedSegment(
                    start=asr_seg.start,
                    end=asr_seg.end,
                    source_text=asr_seg.text.strip(),
                    target_text=target_text,
                    source_language=asr_seg.language or source_language,
                    target_language=target_language,
                    speaker=speaker,
                    confidence=1.0 - asr_seg.no_speech_prob,
                    timestamp=timestamp,
                )
            )

        for seg in fused:
            seg.source_text = normalize_whitespace(seg.source_text)
            seg.source_text = trim_repetitions(seg.source_text)
            seg.target_text = normalize_whitespace(seg.target_text)
            seg.target_text = trim_repetitions(seg.target_text)

        merged = list(merge_short_segments(fused, min_duration=1.0))

        self._pending_segments.extend(merged)

        if merged:
            self._last_emitted_end_time = max(seg.end for seg in merged)

        return merged

    def apply_correction(
        self,
        block_start: float,
        block_end: float,
        corrected_segments: list[TranslatedSegment],
    ) -> None:
        for corrected in corrected_segments:
            corrected.correction_status = "corrected"

        self._pending_segments.sort(key=lambda x: x.timestamp)

        for corrected in corrected_segments:
            matching = None
            for i, pending in enumerate(self._pending_segments):
                if (abs(pending.start - corrected.start) < 0.5 and
                    abs(pending.end - corrected.end) < 0.5):
                    matching = i
                    break
            
            if matching is not None:
                self._pending_segments[matching] = corrected

    def get_all_segments(self) -> list[TranslatedSegment]:
        result = []
        for seg in self._pending_segments:
            result.append(seg)
        return result

    def clear(self) -> None:
        self._pending_segments.clear()
        self._speaker_history.clear()
        self._last_emitted_end_time = 0.0

    def get_segments_in_range(
        self,
        start: float,
        end: float,
    ) -> list[TranslatedSegment]:
        return [seg for seg in self._pending_segments if seg.start >= start and seg.end <= end]

    def export_csv(self, filepath: str) -> None:
        self.export_csv_with_metadata(filepath, {})

    def export_csv_with_metadata(self, filepath: str, metadata: dict) -> None:
        import csv
        import json

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if metadata:
                writer.writerow(["metadata", json.dumps(metadata)])
            writer.writerow(
                [
                    "start",
                    "end",
                    "speaker",
                    "source_text",
                    "target_text",
                    "source_language",
                    "target_language",
                    "confidence",
                    "timestamp",
                ]
            )

            for seg in self._pending_segments:
                writer.writerow(
                    [
                        f"{seg.start:.2f}",
                        f"{seg.end:.2f}",
                        seg.speaker or "UNKNOWN",
                        seg.source_text,
                        seg.target_text,
                        seg.source_language,
                        seg.target_language,
                        f"{seg.confidence:.2f}",
                        f"{seg.timestamp:.2f}",
                    ]
                )

        logger.info(f"Exported {len(self._pending_segments)} segments to CSV: {filepath}")

    def export_jsonl(self, filepath: str) -> None:
        self.export_jsonl_with_metadata(filepath, {})

    def export_jsonl_with_metadata(self, filepath: str, metadata: dict) -> None:
        import json

        with open(filepath, "w", encoding="utf-8") as f:
            if metadata:
                f.write(json.dumps({"metadata": metadata}, ensure_ascii=False) + "\n")
            for seg in self._pending_segments:
                f.write(
                    json.dumps(
                        {
                            "start": seg.start,
                            "end": seg.end,
                            "speaker": seg.speaker,
                            "source_text": seg.source_text,
                            "target_text": seg.target_text,
                            "source_language": seg.source_language,
                            "target_language": seg.target_language,
                            "confidence": seg.confidence,
                            "timestamp": seg.timestamp,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        logger.info(f"Exported {len(self._pending_segments)} segments to JSONL: {filepath}")

    def export_srt(self, filepath: str) -> None:
        self.export_srt_with_metadata(filepath, {})

    def export_srt_with_metadata(self, filepath: str, metadata: dict) -> None:
        import json

        def format_timestamp(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        with open(filepath, "w", encoding="utf-8") as f:
            if metadata:
                f.write(f"# metadata: {json.dumps(metadata, ensure_ascii=False)}\n")
            for i, seg in enumerate(self._pending_segments, 1):
                f.write(f"{i}\n")
                f.write(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n")
                speaker_prefix = f"[{seg.speaker}] " if seg.speaker else ""
                text = seg.target_text if seg.target_text else seg.source_text
                f.write(f"{speaker_prefix}{text}\n\n")

        logger.info(f"Exported {len(self._pending_segments)} segments to SRT: {filepath}")

    def export(self, filepath: str, format: str = "jsonl") -> None:
        format = format.lower()

        if format == "csv":
            self.export_csv(filepath)
        elif format == "jsonl":
            self.export_jsonl(filepath)
        elif format == "srt":
            self.export_srt(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_stats(self) -> dict:
        speakers = {seg.speaker for seg in self._pending_segments if seg.speaker}

        return {
            "total_segments": len(self._pending_segments),
            "unique_speakers": len(speakers),
            "speakers": list(speakers),
        }
