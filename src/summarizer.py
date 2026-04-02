import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from .fusion import TranslatedSegment

logger = logging.getLogger(__name__)

# Common stop words across languages (limited set for broad coverage)
_STOP_WORDS_EN = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "because",
        "if",
        "when",
        "where",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "you",
        "your",
        "yours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
        "about",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "also",
        "like",
        "get",
        "got",
        "go",
        "going",
        "went",
        "come",
        "came",
        "say",
        "said",
        "tell",
        "told",
        "see",
        "saw",
        "know",
        "knew",
        "think",
        "thought",
        "take",
        "took",
        "make",
        "made",
        "use",
        "used",
        "well",
        "really",
        "okay",
        "yeah",
        "yes",
        "right",
        "thing",
        "things",
        "way",
        "actually",
        "basically",
        "mean",
        "want",
        "need",
    }
)


def _tokenize_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristics."""
    if not text:
        return []
    # Split on sentence-ending punctuation followed by whitespace
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    # Also split on double newlines
    sentences = []
    for part in parts:
        sentences.extend(part.split("\n"))
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _tokenize_words(text: str) -> list[str]:
    """Extract words from text, lowercased."""
    return re.findall(r"[\w']+", text.lower())


def _is_cjk(text: str) -> bool:
    """Check if text contains significant CJK characters."""
    cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk_count > len(text) * 0.2


def _segment_cjk_words(text: str) -> list[str]:
    """Rough word segmentation for CJK text using character n-grams."""
    words = []
    i = 0
    while i < len(text):
        ch = text[i]
        if "\u4e00" <= ch <= "\u9fff":
            # Try bigrams for CJK
            if i + 1 < len(text) and "\u4e00" <= text[i + 1] <= "\u9fff":
                words.append(text[i : i + 2].lower())
                words.append(ch.lower())
                i += 2
            else:
                words.append(ch.lower())
                i += 1
        elif ch.isalnum() or ch in ("'", "_"):
            # Accumulate latin words
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] in ("'", "_")):
                j += 1
            words.append(text[i:j].lower())
            i = j
        else:
            i += 1
    return [w for w in words if len(w) > 1]


@dataclass
class ScoredSentence:
    text: str
    score: float
    original_index: int
    speaker: Optional[str] = None
    start: float = 0.0
    end: float = 0.0


class MeetingSummarizer:
    """Extractive meeting summarizer using word frequency and position scoring.

    No external LLM dependency. Works on CPU.
    """

    def __init__(
        self,
        key_point_ratio: float = 0.3,
        min_sentence_words: int = 3,
        position_weight: float = 0.15,
        length_weight: float = 0.1,
    ):
        """
        Args:
            key_point_ratio: Fraction of sentences to extract as key points.
            min_sentence_words: Minimum words in a sentence to consider it.
            position_weight: Weight for sentence position score (earlier = higher).
            length_weight: Weight for sentence length preference (moderate length preferred).
        """
        self.key_point_ratio = key_point_ratio
        self.min_sentence_words = min_sentence_words
        self.position_weight = position_weight
        self.length_weight = length_weight

    def _get_word_frequencies(self, texts: list[str]) -> Counter:
        """Build word frequency table from all texts."""
        freq = Counter()
        for text in texts:
            if _is_cjk(text):
                words = _segment_cjk_words(text)
            else:
                words = _tokenize_words(text)
            for w in words:
                if w not in _STOP_WORDS_EN and len(w) > 1:
                    freq[w] += 1
        return freq

    def _score_sentence(
        self,
        sentence: str,
        word_freq: Counter,
        max_freq: float,
        position: int,
        total_sentences: int,
    ) -> float:
        """Score a sentence based on word frequency, position, and length."""
        if _is_cjk(sentence):
            words = _segment_cjk_words(sentence)
        else:
            words = _tokenize_words(sentence)

        if not words:
            return 0.0

        # Word frequency score (normalized)
        freq_score = sum(word_freq.get(w, 0) for w in words)
        freq_score = freq_score / (len(words) * max_freq) if max_freq > 0 else 0.0

        # Position score: first and last sentences get higher scores
        if total_sentences <= 1:
            pos_score = 1.0
        else:
            normalized_pos = position / (total_sentences - 1)
            # U-shape: boost first 20% and last 20% of the transcript
            pos_score = max(
                1.0 - normalized_pos * 2,  # Linear decay from start
                normalized_pos * 1.5 - 0.5,  # Linear rise to end
                0.3,  # Minimum floor
            )

        # Length score: prefer moderate length (not too short, not too long)
        word_count = len(words)
        if word_count < 5:
            length_score = 0.3
        elif word_count < 20:
            length_score = 0.8
        elif word_count < 40:
            length_score = 1.0
        else:
            length_score = 0.7

        # Speaker presence bonus
        has_speaker = any(
            marker in sentence.lower()
            for marker in ["said", "mentioned", "proposed", "agreed", "noted"]
        )
        speaker_bonus = 0.1 if has_speaker else 0.0

        return (
            freq_score * (1 - self.position_weight - self.length_weight)
            + pos_score * self.position_weight
            + length_score * self.length_weight
            + speaker_bonus
        )

    def _extract_key_points(
        self, segments: list[TranslatedSegment], use_target: bool
    ) -> list[str]:
        """Extract key points from the transcript."""
        # Choose text field based on language preference
        if use_target:
            texts = [
                seg.target_text or seg.source_text
                for seg in segments
                if seg.target_text or seg.source_text
            ]
        else:
            texts = [seg.source_text for seg in segments if seg.source_text]

        if not texts:
            return []

        full_text = " ".join(texts)
        sentences = _tokenize_sentences(full_text)

        if not sentences:
            # Fall back to treating each segment as a sentence
            sentences = [t for t in texts if len(t.strip()) > 5]

        if not sentences:
            return []

        word_freq = self._get_word_frequencies(texts)
        max_freq = max(word_freq.values()) if word_freq else 1.0

        scored = []
        for i, sent in enumerate(sentences):
            score = self._score_sentence(sent, word_freq, max_freq, i, len(sentences))
            scored.append(ScoredSentence(text=sent, score=score, original_index=i))

        scored.sort(key=lambda s: s.score, reverse=True)

        # Take top N sentences, then re-sort by original position
        n = max(1, int(len(sentences) * self.key_point_ratio))
        n = min(n, len(scored))
        top = sorted(scored[:n], key=lambda s: s.original_index)

        return [s.text for s in top]

    def _build_speaker_summary(
        self, segments: list[TranslatedSegment]
    ) -> dict[str, dict]:
        """Build per-speaker contribution summary."""
        speaker_data = {}
        for seg in segments:
            speaker = seg.speaker or "Unknown"
            if speaker not in speaker_data:
                speaker_data[speaker] = {
                    "duration": 0.0,
                    "segments": 0,
                    "word_count": 0,
                    "texts": [],
                }
            duration = seg.end - seg.start
            speaker_data[speaker]["duration"] += duration
            speaker_data[speaker]["segments"] += 1
            text = seg.target_text or seg.source_text or ""
            speaker_data[speaker]["word_count"] += len(text.split())
            if text:
                speaker_data[speaker]["texts"].append(text)

        return speaker_data

    def _detect_topic_boundaries(
        self, segments: list[TranslatedSegment], use_target: bool
    ) -> list[tuple[float, float, str]]:
        """Detect topic segments using text similarity between consecutive windows."""
        if len(segments) < 4:
            if segments:
                first_text = ""
                for seg in segments:
                    t = seg.target_text if use_target and seg.target_text else seg.source_text
                    if t:
                        first_text += t + " "
                return [
                    (segments[0].start, segments[-1].end, first_text.strip()[:80])
                ]
            return []

        # Build word sets per segment
        def get_words(seg):
            text = (
                seg.target_text if use_target and seg.target_text else seg.source_text
            )
            if _is_cjk(text):
                return set(_segment_cjk_words(text))
            return set(_tokenize_words(text))

        segment_words = [get_words(seg) for seg in segments]

        # Compute similarity between adjacent windows (3-segment windows)
        window_size = 3
        topics = []
        current_start = 0

        for i in range(window_size, len(segment_words)):
            prev_window = set()
            for j in range(max(0, i - window_size), i):
                prev_window |= segment_words[j]

            curr_window = segment_words[i]

            if prev_window and curr_window:
                similarity = len(prev_window & curr_window) / len(
                    prev_window | curr_window
                )
            else:
                similarity = 1.0

            if similarity < 0.15 and i - current_start >= 2:
                # Topic boundary detected
                topic_text = " ".join(
                    seg.target_text or seg.source_text
                    for seg in segments[current_start : i + 1]
                    if seg.target_text or seg.source_text
                )
                topics.append(
                    (
                        segments[current_start].start,
                        segments[min(i, len(segments) - 1)].end,
                        topic_text[:100].strip(),
                    )
                )
                current_start = i

        # Final topic
        if current_start < len(segments):
            topic_text = " ".join(
                seg.target_text or seg.source_text
                for seg in segments[current_start:]
                if seg.target_text or seg.source_text
            )
            topics.append(
                (
                    segments[current_start].start,
                    segments[-1].end,
                    topic_text[:100].strip(),
                )
            )

        return topics

    def summarize(
        self,
        segments: list[TranslatedSegment],
        prefer_target_language: bool = True,
    ) -> dict:
        """Summarize meeting transcript from translated segments.

        Args:
            segments: List of TranslatedSegment objects.
            prefer_target_language: If True, prefer target_text for summaries.

        Returns:
            Dict with keys:
                - 'key_points': list of str (extracted important sentences)
                - 'speaker_summary': dict mapping speaker names to contribution stats
                - 'full_summary': str (complete extractive summary)
        """
        if not segments:
            return {
                "key_points": [],
                "speaker_summary": {},
                "full_summary": "",
            }

        use_target = prefer_target_language and any(
            seg.target_text for seg in segments
        )

        # Extract key points
        key_points = self._extract_key_points(segments, use_target)

        # Build speaker summary
        speaker_data = self._build_speaker_summary(segments)
        speaker_summary = {}
        for speaker, data in speaker_data.items():
            total_duration = segments[-1].end - segments[0].start
            pct = (
                (data["duration"] / total_duration * 100) if total_duration > 0 else 0
            )
            speaker_summary[speaker] = {
                "speaking_time_seconds": round(data["duration"], 1),
                "contribution_percent": round(pct, 1),
                "segment_count": data["segments"],
                "word_count": data["word_count"],
            }

        # Build full summary (concatenate key points with topic segments)
        topic_segments = self._detect_topic_boundaries(segments, use_target)
        full_summary_parts = []

        if key_points:
            full_summary_parts.append("## Key Points")
            for point in key_points:
                full_summary_parts.append(f"- {point}")

        if speaker_summary:
            full_summary_parts.append("\n## Speaker Contributions")
            for speaker, stats in speaker_summary.items():
                full_summary_parts.append(
                    f"- **{speaker}**: {stats['contribution_percent']}% speaking time "
                    f"({stats['segment_count']} segments, {stats['word_count']} words)"
                )

        if topic_segments and len(topic_segments) > 1:
            full_summary_parts.append("\n## Topics Discussed")
            for i, (start, end, preview) in enumerate(topic_segments, 1):
                start_ts = self._format_timestamp(start)
                end_ts = self._format_timestamp(end)
                full_summary_parts.append(
                    f"- **Topic {i}** ({start_ts} - {end_ts}): {preview}..."
                )

        full_summary = "\n".join(full_summary_parts)

        return {
            "key_points": key_points,
            "speaker_summary": speaker_summary,
            "full_summary": full_summary,
        }

    def export_markdown(
        self,
        segments: list[TranslatedSegment],
        prefer_target_language: bool = True,
    ) -> str:
        """Export meeting summary as markdown document.

        Returns:
            Markdown-formatted meeting summary.
        """
        result = self.summarize(segments, prefer_target_language)

        lines = []
        lines.append("# Meeting Summary\n")

        # Key points
        if result["key_points"]:
            lines.append("## Key Points\n")
            for point in result["key_points"]:
                lines.append(f"- {point}")
            lines.append("")

        # Speaker contributions
        if result["speaker_summary"]:
            lines.append("## Speaker Contributions\n")
            lines.append("| Speaker | Speaking Time | Contribution | Segments | Words |")
            lines.append("|---------|--------------|-------------|----------|-------|")
            for speaker, stats in result["speaker_summary"].items():
                lines.append(
                    f"| {speaker} | "
                    f"{stats['speaking_time_seconds']}s | "
                    f"{stats['contribution_percent']}% | "
                    f"{stats['segment_count']} | "
                    f"{stats['word_count']} |"
                )
            lines.append("")

        # Full transcript summary
        lines.append("## Full Summary\n")
        if result["full_summary"]:
            lines.append(result["full_summary"])
        lines.append("")

        # Transcript with speaker labels
        lines.append("## Transcript\n")
        for seg in segments:
            speaker_label = seg.speaker or "Unknown"
            text = (
                seg.target_text
                if prefer_target_language and seg.target_text
                else seg.source_text
            )
            timestamp = self._format_timestamp(seg.start)
            lines.append(f"**[{timestamp}] {speaker_label}:** {text}")

        return "\n".join(lines)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
