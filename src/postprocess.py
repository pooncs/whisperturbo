import re
from typing import List


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def trim_repetitions(text: str) -> str:
    words = text.split()
    if not words:
        return text

    result = [words[0]]
    for i in range(1, len(words)):
        if words[i].lower() != result[-1].lower():
            result.append(words[i])

    return " ".join(result)


def merge_short_segments(segments: List, min_duration: float = 1.0) -> List:
    if not segments:
        return segments

    if len(segments) == 1:
        return segments

    merged = [segments[0]]

    for i in range(1, len(segments)):
        current = segments[i]
        previous = merged[-1]

        time_gap = current.start - previous.end
        duration = current.end - current.start

        if (
            current.speaker == previous.speaker
            and time_gap < 0.5
            and duration < min_duration
        ):
            previous.text = previous.text + " " + current.text
            previous.end = current.end
        else:
            merged.append(current)

    return merged
