import time
from typing import List, Optional, Dict
from dataclasses import dataclass
import logging

from .whisper_asr import TranscriptionSegment
from .diarization import SpeakerSegment


logger = logging.getLogger(__name__)


@dataclass
class TranslatedSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str]
    language: str
    confidence: float = 1.0
    timestamp: float = 0.0


class Fusion:
    def __init__(self, min_overlap: float = 0.3):
        self.min_overlap = min_overlap
        self._segments: List[TranslatedSegment] = []
        self._speaker_history: Dict[str, float] = {}
        
    def _calculate_overlap(self, asr_seg: TranscriptionSegment, speaker_seg: SpeakerSegment) -> float:
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
        speaker_segments: List[SpeakerSegment],
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
        asr_segments: List[TranscriptionSegment],
        speaker_segments: List[SpeakerSegment],
        timestamp: float,
    ) -> List[TranslatedSegment]:
        fused = []
        
        for asr_seg in asr_segments:
            if not asr_seg.text:
                continue
            
            speaker = self._get_dominant_speaker(asr_seg, speaker_segments)
            
            if speaker is None:
                speaker = self._get_last_speaker()
            
            fused.append(TranslatedSegment(
                start=asr_seg.start,
                end=asr_seg.end,
                text=asr_seg.text,
                speaker=speaker,
                language=asr_seg.language,
                confidence=1.0 - asr_seg.no_speech_prob,
                timestamp=timestamp,
            ))
        
        self._segments.extend(fused)
        return fused
    
    def get_all_segments(self) -> List[TranslatedSegment]:
        return self._segments.copy()
    
    def clear(self) -> None:
        self._segments.clear()
        self._speaker_history.clear()
    
    def get_segments_in_range(
        self,
        start: float,
        end: float,
    ) -> List[TranslatedSegment]:
        return [
            seg for seg in self._segments
            if seg.start >= start and seg.end <= end
        ]
    
    def export_csv(self, filepath: str) -> None:
        import csv
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['start', 'end', 'speaker', 'text', 'language', 'confidence', 'timestamp'])
            
            for seg in self._segments:
                writer.writerow([
                    f"{seg.start:.2f}",
                    f"{seg.end:.2f}",
                    seg.speaker or "UNKNOWN",
                    seg.text,
                    seg.language,
                    f"{seg.confidence:.2f}",
                    f"{seg.timestamp:.2f}",
                ])
        
        logger.info(f"Exported {len(self._segments)} segments to CSV: {filepath}")
    
    def export_jsonl(self, filepath: str) -> None:
        import json
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for seg in self._segments:
                f.write(json.dumps({
                    'start': seg.start,
                    'end': seg.end,
                    'speaker': seg.speaker,
                    'text': seg.text,
                    'language': seg.language,
                    'confidence': seg.confidence,
                    'timestamp': seg.timestamp,
                }, ensure_ascii=False) + '\n')
        
        logger.info(f"Exported {len(self._segments)} segments to JSONL: {filepath}")
    
    def export_srt(self, filepath: str) -> None:
        def format_timestamp(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(self._segments, 1):
                f.write(f"{i}\n")
                f.write(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n")
                speaker_prefix = f"[{seg.speaker}] " if seg.speaker else ""
                f.write(f"{speaker_prefix}{seg.text}\n\n")
        
        logger.info(f"Exported {len(self._segments)} segments to SRT: {filepath}")
    
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
        speakers = set(seg.speaker for seg in self._segments if seg.speaker)
        
        return {
            "total_segments": len(self._segments),
            "unique_speakers": len(speakers),
            "speakers": list(speakers),
        }
