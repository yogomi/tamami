"""Tests for the speech recognition module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.speech.recognition import WhisperRecognizer


class TestWhisperRecognizer:
    """Tests for the WhisperRecognizer class."""

    @patch("src.speech.recognition.whisper.load_model")
    def test_init_valid_model(self, mock_load_model):
        """Test initialization with a valid model name."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        WhisperRecognizer(model_name="base")

        mock_load_model.assert_called_once_with("base")

    def test_init_invalid_model_raises(self):
        """Test initialization with an invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model name"):
            WhisperRecognizer(model_name="invalid_model")

    @patch("src.speech.recognition.whisper.load_model")
    def test_transcribe_basic(self, mock_load_model):
        """Test basic transcription."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_model.transcribe.return_value = {
            "text": "Hello world",
            "language": "en",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello world"}],
        }

        recognizer = WhisperRecognizer()
        audio = np.zeros(16000, dtype=np.float32)
        result = recognizer.transcribe(audio)

        assert result["text"] == "Hello world"
        assert result["language"] == "en"
        assert len(result["segments"]) == 1

    @patch("src.speech.recognition.whisper.load_model")
    def test_transcribe_with_language(self, mock_load_model):
        """Test transcription with specified language."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_model.transcribe.return_value = {
            "text": "こんにちは",
            "language": "ja",
            "segments": [],
        }

        recognizer = WhisperRecognizer()
        audio = np.zeros(16000, dtype=np.float32)
        recognizer.transcribe(audio, language="ja")

        mock_model.transcribe.assert_called_once()
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs.get("language") == "ja"

    @patch("src.speech.recognition.whisper.load_model")
    def test_transcribe_converts_dtype(self, mock_load_model):
        """Test transcription converts audio to float32."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_model.transcribe.return_value = {
            "text": "",
            "language": "en",
            "segments": [],
        }

        recognizer = WhisperRecognizer()
        # Pass int16 audio
        audio = np.zeros(16000, dtype=np.int16)
        recognizer.transcribe(audio)

        # Verify transcribe was called with float32
        call_args = mock_model.transcribe.call_args[0]
        assert call_args[0].dtype == np.float32

    @patch("src.speech.recognition.whisper.load_model")
    @patch("src.speech.recognition.whisper.pad_or_trim")
    @patch("src.speech.recognition.whisper.log_mel_spectrogram")
    def test_detect_language(self, mock_log_mel, mock_pad_trim, mock_load_model):
        """Test language detection."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_model.dims.n_mels = 80
        mock_model.device = "cpu"

        mock_mel = MagicMock()
        mock_mel.to.return_value = mock_mel
        mock_log_mel.return_value = mock_mel

        mock_pad_trim.return_value = np.zeros(16000 * 30, dtype=np.float32)
        mock_model.detect_language.return_value = (None, {"en": 0.9, "ja": 0.1})

        recognizer = WhisperRecognizer()
        audio = np.zeros(16000, dtype=np.float32)
        language = recognizer.detect_language(audio)

        assert language == "en"

    @patch("src.speech.recognition.whisper.load_model")
    def test_valid_models(self, mock_load_model):
        """Test all valid model names."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        for model_name in ["tiny", "base", "small", "medium", "large"]:
            WhisperRecognizer(model_name=model_name)
            mock_load_model.assert_called_with(model_name)
