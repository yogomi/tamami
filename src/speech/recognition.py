"""Speech recognition module using OpenAI Whisper.

This module provides speech-to-text functionality using the Whisper model.
"""

from typing import Dict, List, Optional, Any

import numpy as np
import whisper


class WhisperRecognizer:
    """Speech recognizer using OpenAI Whisper model.

    This class provides speech-to-text conversion using Whisper,
    supporting multiple model sizes and automatic language detection.

    Args:
        model_name: Name of the Whisper model to use.
            Options: "tiny", "base", "small", "medium", "large".
            Defaults to "base".

    Examples:
        >>> recognizer = WhisperRecognizer(model_name="base")
        >>> audio_data = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        >>> result = recognizer.transcribe(audio_data)
        >>> print(result["text"])
    """

    VALID_MODELS = ["tiny", "base", "small", "medium", "large"]

    def __init__(self, model_name: str = "base") -> None:
        """Initialize the Whisper recognizer.

        Args:
            model_name: Name of the Whisper model to use.
                Options: "tiny", "base", "small", "medium", "large".
                Defaults to "base".

        Raises:
            ValueError: If model_name is not a valid Whisper model.

        Examples:
            >>> recognizer = WhisperRecognizer(model_name="tiny")
            >>> recognizer = WhisperRecognizer(model_name="base")
        """
        if model_name not in self.VALID_MODELS:
            raise ValueError(
                f"Invalid model name: {model_name}. "
                f"Valid options: {', '.join(self.VALID_MODELS)}"
            )
        self._model_name = model_name
        self._model = whisper.load_model(model_name)

    def transcribe(
        self,
        audio_data: np.ndarray,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe audio data to text.

        Args:
            audio_data: Audio data as a numpy array with float32 dtype.
                Expected sample rate is 16000Hz.
            language: Language code for transcription (e.g., "en", "ja", "uk").
                If None, language is automatically detected.

        Returns:
            Dictionary containing:
                - text: Transcribed text as string.
                - language: Detected or specified language code.
                - segments: List of segment dictionaries with timing info.

        Examples:
            >>> recognizer = WhisperRecognizer()
            >>> audio = np.random.randn(16000).astype(np.float32) * 0.1
            >>> result = recognizer.transcribe(audio)
            >>> print(result["text"])
            >>> print(result["language"])

            >>> # With explicit language
            >>> result = recognizer.transcribe(audio, language="ja")
            >>> print(result["language"])
            'ja'
        """
        # Ensure audio is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Prepare transcription options
        options: Dict[str, Any] = {}
        if language is not None:
            options["language"] = language

        # Run transcription
        result = self._model.transcribe(audio_data, **options)

        # Extract segments
        segments: List[Dict[str, Any]] = []
        for segment in result.get("segments", []):
            segments.append(
                {
                    "id": segment.get("id"),
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": segment.get("text"),
                }
            )

        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", language or "unknown"),
            "segments": segments,
        }

    def detect_language(self, audio_data: np.ndarray) -> str:
        """Detect the language of the audio.

        Args:
            audio_data: Audio data as a numpy array with float32 dtype.
                Expected sample rate is 16000Hz.

        Returns:
            Detected language code (e.g., "en", "ja", "uk").

        Examples:
            >>> recognizer = WhisperRecognizer()
            >>> audio = np.random.randn(16000).astype(np.float32) * 0.1
            >>> language = recognizer.detect_language(audio)
            >>> print(language)
        """
        # Ensure audio is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Pad or trim audio to 30 seconds for language detection
        audio_padded = whisper.pad_or_trim(audio_data)

        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio_padded, n_mels=self._model.dims.n_mels)
        mel = mel.to(self._model.device)

        # Detect language
        _, probs = self._model.detect_language(mel)
        detected_language = max(probs, key=probs.get)

        return detected_language
