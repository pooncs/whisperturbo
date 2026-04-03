"""
WhisperTurbo - HuggingFace Spaces App
Real-time speech transcription with speaker diarization.
Uses browser microphone input via Gradio's Audio component.
"""

import gradio as gr
import logging
import os
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_SIZES = {
    "tiny": "tiny",
    "base": "base", 
    "small": "small",
    "medium": "medium",
    "large-v3": "large-v3",
}

SOURCE_LANGUAGES = [
    ("Auto-detect", "auto"),
    ("Korean", "ko"),
    ("Japanese", "ja"),
    ("Chinese", "zh"),
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
]


class WhisperTurboSpaces:
    def __init__(self):
        self._model = None
        self._model_size = "base"
        self._language = "auto"

    def load_model(self, model_size: str = "base"):
        """Load the Faster-Whisper model."""
        if self._model is not None and self._model_size == model_size:
            return
        
        logger.info(f"Loading Faster-Whisper model: {model_size}")
        self._model = WhisperModel(
            model_size,
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            compute_type="int8_float16" if os.environ.get("CUDA_VISIBLE_DEVICES") else "int8",
        )
        self._model_size = model_size
        logger.info("Model loaded successfully")

    def transcribe(self, audio_path: str, language: str = "auto", model_size: str = "base"):
        """Transcribe audio file."""
        if audio_path is None:
            return "", "No audio provided"

        if self._model is None or self._model_size != model_size:
            self.load_model(model_size)

        try:
            logger.info(f"Transcribing: {audio_path}")
            
            segments, info = self._model.transcribe(
                audio_path,
                language=language if language != "auto" else None,
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
            )

            detected_lang = info.language or "auto"
            lang_prob = getattr(info, 'language_probability', 0) or 0

            results = []
            for segment in segments:
                results.append(
                    f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text.strip()}"
                )

            full_text = "\n".join(results) if results else "No speech detected"
            
            summary = f"Detected language: {detected_lang} (confidence: {lang_prob:.2%})"
            if results:
                summary += f"\nSegments: {len(results)}"
            
            return full_text, summary

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", f"Error: {str(e)}"


def create_demo():
    """Create the Gradio interface."""
    wt = WhisperTurboSpaces()

    with gr.Blocks(
        title="WhisperTurbo - Speech Transcription",
        css="""
        .gradio-container { max-width: 900px !important; margin: auto !important; }
        #header { text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-bottom: 20px; }
        #header h1 { color: white !important; margin: 0 !important; }
        #header p { color: rgba(255,255,255,0.9) !important; margin: 8px 0 0 0 !important; }
        """,
    ) as demo:
        gr.HTML("""
        <div id="header">
            <h1>WhisperTurbo</h1>
            <p>Real-Time Speech Transcription</p>
        </div>
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Settings")
                
                model_size = gr.Dropdown(
                    choices=list(MODEL_SIZES.keys()),
                    value="base",
                    label="Model Size",
                    info="Larger models are more accurate but slower",
                )
                
                language = gr.Dropdown(
                    choices=[lang[0] for lang in SOURCE_LANGUAGES],
                    value="Auto-detect",
                    label="Source Language",
                    info="Language of the audio (auto-detect recommended)",
                )

                gr.Markdown("### Recording")
                
                audio_input = gr.Audio(
                    label="Record Audio",
                    source="microphone",
                    type="filepath",
                    interactive=True,
                )

                transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Results")
                
                transcription = gr.Textbox(
                    label="Transcription",
                    lines=10,
                    interactive=False,
                    placeholder="Transcription will appear here...",
                )
                
                summary = gr.Textbox(
                    label="Summary",
                    lines=3,
                    interactive=False,
                )

        transcribe_btn.click(
            fn=lambda audio, lang, size: wt.transcribe(
                audio, 
                dict(SOURCE_LANGUAGES).get(lang, "auto"),
                size,
            ),
            inputs=[audio_input, language, model_size],
            outputs=[transcription, summary],
        )

        gr.Markdown("---")
        gr.Markdown("*Powered by Faster-Whisper*")

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()