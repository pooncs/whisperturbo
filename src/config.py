import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Language definitions for Whisper-compatible ASR
# ---------------------------------------------------------------------------

# Source languages that Whisper can transcribe. Each entry is (code, name).
# Whisper supports ~99 languages; this list covers all languages that
# faster-whisper / OpenAI Whisper models can detect and transcribe.
SUPPORTED_SOURCE_LANGUAGES: list[dict[str, str]] = [
    {"code": "auto",  "name": "Auto-detect"},
    {"code": "en",    "name": "English"},
    {"code": "zh",    "name": "Chinese"},
    {"code": "de",    "name": "German"},
    {"code": "es",    "name": "Spanish"},
    {"code": "ru",    "name": "Russian"},
    {"code": "ko",    "name": "Korean"},
    {"code": "fr",    "name": "French"},
    {"code": "ja",    "name": "Japanese"},
    {"code": "pt",    "name": "Portuguese"},
    {"code": "tr",    "name": "Turkish"},
    {"code": "pl",    "name": "Polish"},
    {"code": "ca",    "name": "Catalan"},
    {"code": "nl",    "name": "Dutch"},
    {"code": "ar",    "name": "Arabic"},
    {"code": "sv",    "name": "Swedish"},
    {"code": "it",    "name": "Italian"},
    {"code": "id",    "name": "Indonesian"},
    {"code": "hi",    "name": "Hindi"},
    {"code": "fi",    "name": "Finnish"},
    {"code": "vi",    "name": "Vietnamese"},
    {"code": "he",    "name": "Hebrew"},
    {"code": "uk",    "name": "Ukrainian"},
    {"code": "el",    "name": "Greek"},
    {"code": "ms",    "name": "Malay"},
    {"code": "cs",    "name": "Czech"},
    {"code": "ro",    "name": "Romanian"},
    {"code": "da",    "name": "Danish"},
    {"code": "hu",    "name": "Hungarian"},
    {"code": "ta",    "name": "Tamil"},
    {"code": "no",    "name": "Norwegian"},
    {"code": "th",    "name": "Thai"},
    {"code": "ur",    "name": "Urdu"},
    {"code": "hr",    "name": "Croatian"},
    {"code": "bg",    "name": "Bulgarian"},
    {"code": "lt",    "name": "Lithuanian"},
    {"code": "la",    "name": "Latin"},
    {"code": "mi",    "name": "Maori"},
    {"code": "ml",    "name": "Malayalam"},
    {"code": "cy",    "name": "Welsh"},
    {"code": "sk",    "name": "Slovak"},
    {"code": "te",    "name": "Telugu"},
    {"code": "fa",    "name": "Persian"},
    {"code": "lv",    "name": "Latvian"},
    {"code": "bn",    "name": "Bengali"},
    {"code": "sr",    "name": "Serbian"},
    {"code": "az",    "name": "Azerbaijani"},
    {"code": "sl",    "name": "Slovenian"},
    {"code": "kn",    "name": "Kannada"},
    {"code": "et",    "name": "Estonian"},
    {"code": "mk",    "name": "Macedonian"},
    {"code": "br",    "name": "Breton"},
    {"code": "eu",    "name": "Basque"},
    {"code": "is",    "name": "Icelandic"},
    {"code": "hy",    "name": "Armenian"},
    {"code": "ne",    "name": "Nepali"},
    {"code": "mn",    "name": "Mongolian"},
    {"code": "bs",    "name": "Bosnian"},
    {"code": "kk",    "name": "Kazakh"},
    {"code": "sq",    "name": "Albanian"},
    {"code": "sw",    "name": "Swahili"},
    {"code": "gl",    "name": "Galician"},
    {"code": "mr",    "name": "Marathi"},
    {"code": "pa",    "name": "Punjabi"},
    {"code": "si",    "name": "Sinhala"},
    {"code": "km",    "name": "Khmer"},
    {"code": "sn",    "name": "Shona"},
    {"code": "yo",    "name": "Yoruba"},
    {"code": "so",    "name": "Somali"},
    {"code": "af",    "name": "Afrikaans"},
    {"code": "oc",    "name": "Occitan"},
    {"code": "ka",    "name": "Georgian"},
    {"code": "be",    "name": "Belarusian"},
    {"code": "tg",    "name": "Tajik"},
    {"code": "sd",    "name": "Sindhi"},
    {"code": "gu",    "name": "Gujarati"},
    {"code": "am",    "name": "Amharic"},
    {"code": "yi",    "name": "Yiddish"},
    {"code": "lo",    "name": "Lao"},
    {"code": "uz",    "name": "Uzbek"},
    {"code": "fo",    "name": "Faroese"},
    {"code": "ht",    "name": "Haitian Creole"},
    {"code": "ps",    "name": "Pashto"},
    {"code": "tk",    "name": "Turkmen"},
    {"code": "nn",    "name": "Nynorsk"},
    {"code": "mt",    "name": "Maltese"},
    {"code": "sa",    "name": "Sanskrit"},
    {"code": "lb",    "name": "Luxembourgish"},
    {"code": "my",    "name": "Myanmar"},
    {"code": "bo",    "name": "Tibetan"},
    {"code": "tl",    "name": "Tagalog"},
    {"code": "mg",    "name": "Malagasy"},
    {"code": "as",    "name": "Assamese"},
    {"code": "tt",    "name": "Tatar"},
    {"code": "haw",   "name": "Hawaiian"},
    {"code": "ln",    "name": "Lingala"},
    {"code": "ha",    "name": "Hausa"},
    {"code": "ba",    "name": "Bashkir"},
    {"code": "jw",    "name": "Javanese"},
    {"code": "su",    "name": "Sundanese"},
    {"code": "yue",   "name": "Cantonese"},
]

# Target languages: Whisper's translate task outputs English only, but we can
# attempt transcription-output translation via initial_prompt tricks.
# All source languages (except "auto") are valid targets.
SUPPORTED_TARGET_LANGUAGES: list[dict[str, str]] = [
    lang for lang in SUPPORTED_SOURCE_LANGUAGES if lang["code"] != "auto"
]

# ---------------------------------------------------------------------------
# Mapping utilities
# ---------------------------------------------------------------------------

# code → name  (e.g. "ko" → "Korean")
LANGUAGE_CODE_TO_NAME: dict[str, str] = {
    lang["code"]: lang["name"] for lang in SUPPORTED_SOURCE_LANGUAGES
}

# name (lowercased) → code  (e.g. "korean" → "ko")
LANGUAGE_NAME_TO_CODE: dict[str, str] = {
    lang["name"].lower(): lang["code"] for lang in SUPPORTED_SOURCE_LANGUAGES
}

# code → name for targets only
TARGET_LANGUAGE_CODE_TO_NAME: dict[str, str] = {
    lang["code"]: lang["name"] for lang in SUPPORTED_TARGET_LANGUAGES
}


def language_name_to_code(name: str) -> str | None:
    """Convert a language name (case-insensitive) to its Whisper code.

    Returns None if the name is not recognised.
    """
    return LANGUAGE_NAME_TO_CODE.get(name.lower().strip())


def language_code_to_name(code: str) -> str | None:
    """Convert a Whisper language code to its display name.

    Returns None if the code is not recognised.
    """
    return LANGUAGE_CODE_TO_NAME.get(code)


def validate_language_pair(source_code: str, target_code: str) -> list[str]:
    """Validate a source/target language pair and return warning messages.

    An empty list means the pair is valid.
    """
    warnings: list[str] = []
    if source_code != "auto" and source_code not in LANGUAGE_CODE_TO_NAME:
        warnings.append(f"Unknown source language code: {source_code!r}")
    if target_code not in TARGET_LANGUAGE_CODE_TO_NAME:
        warnings.append(f"Unknown target language code: {target_code!r}")
    if (
        source_code != "auto"
        and source_code != "en"
        and target_code != "en"
    ):
        warnings.append(
            f"Whisper's translate task only outputs English. "
            f"Non-English target ({target_code}) uses experimental initial_prompt workaround."
        )
    return warnings


def _detect_device() -> str:
    """Auto-detect CUDA availability, default to CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _default_compute_type() -> str:
    """Return appropriate compute_type for the detected device."""
    if _detect_device() == "cuda":
        return "float16"
    return "int8"


@dataclass
class Config:
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    DTYPE: str = "float32"
    CHUNK_DURATION: float = 0.1
    BUFFER_DURATION: float = 30.0

    # Layered transcription models
    WHISPER_FAST_MODEL: str = "base"          # fast model for real-time (base or small)
    WHISPER_CORRECT_MODEL: str = "large-v3-turbo"  # correction model for block passes

    # Legacy single-model field (kept for backward compat, points to correct model)
    WHISPER_MODEL: str = "large-v3-turbo"

    WHISPER_DEVICE: str = field(default_factory=_detect_device)
    WHISPER_COMPUTE_TYPE: str = field(default_factory=_default_compute_type)
    WHISPER_TASK: str = "transcribe"  # transcribe in source language first, then translate
    WHISPER_LANGUAGE: str = "auto"  # auto for automatic language detection
    WHISPER_TARGET_LANGUAGE: str = "en"  # default target language for translation

    WHISPER_BEAM_SIZE: int = 5
    WHISPER_NO_SPEECH_THRESHOLD: float = 0.6
    WHISPER_LOGPROB_THRESHOLD: float = -1.0
    WHISPER_COMPRESSION_RATIO_THRESHOLD: float = 2.4
    WHISPER_INITIAL_PROMPT: str = ""

    MIN_PROCESSING_INTERVAL: float = 2.0

    # Correction pass settings
    CORRECTION_BLOCK_DURATION: float = 5.0  # seconds of audio per correction block

    # Summarization
    SUMMARY_ENABLED: bool = True

    ENABLE_CONTEXT_CARRY: bool = True
    CONTEXT_MAX_LENGTH: int = 500
    CONTEXT_SEGMENT_COUNT: int = 3

    # PyAnnote diarization model - community version is free but requires HF_TOKEN
    # For production, consider pyannote/speaker-diarization-3.1 (gated, requires acceptance)
    DIARIZATION_MODEL: str = "pyannote/speaker-diarization-community-1"
    DIARIZATION_WINDOW_SIZE: float = 15.0
    DIARIZATION_OVERLAP: float = 5.0

    # Pyannote segmentation model (used by diarization pipeline)
    DIARIZATION_SEGMENTATION_MODEL: str = "pyannote/segmentation-3.0"

    VAD_THRESHOLD: float = 0.3
    VAD_MIN_SPEECH_DURATION: float = 0.1
    VAD_MIN_SILENCE_DURATION: float = 0.3

    GUI_REFRESH_RATE: int = 100
    GUI_MAX_ROWS: int = 100

    EXPORT_FORMATS: tuple = ("csv", "jsonl", "srt")

    SPEAKER_COLORS: tuple = (
        "#FF0000",
        "#00FF00",
        "#0000FF",
        "#FFFF00",
        "#FF00FF",
        "#00FFFF",
        "#FFA500",
        "#800080",
        "#008080",
        "#FFC0CB",
        "#A52A2A",
        "#808080",
    )

    @property
    def hf_token(self) -> str:
        return os.environ.get("HF_TOKEN", "")

    @property
    def buffer_size(self) -> int:
        return int(self.SAMPLE_RATE * self.BUFFER_DURATION)

    @property
    def chunk_samples(self) -> int:
        return int(self.SAMPLE_RATE * self.CHUNK_DURATION)


CONFIG = Config()
