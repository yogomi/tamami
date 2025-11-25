"""Tests for the main module."""

from unittest.mock import MagicMock, patch

import pytest

from main import create_audio_input, parse_args


class TestParseArgs:
    """Tests for command-line argument parsing."""

    def test_default_args(self):
        """Test default argument values."""
        with patch("sys.argv", ["main.py"]):
            args = parse_args()
            assert args.input_source == "microphone"
            assert args.model == "base"
            assert args.language is None
            assert args.chunk_duration == 5.0

    def test_custom_args(self):
        """Test custom argument values."""
        with patch(
            "sys.argv",
            [
                "main.py",
                "--input-source",
                "stream",
                "--model",
                "tiny",
                "--language",
                "ja",
                "--chunk-duration",
                "10",
            ],
        ):
            args = parse_args()
            assert args.input_source == "stream"
            assert args.model == "tiny"
            assert args.language == "ja"
            assert args.chunk_duration == 10.0


class TestCreateAudioInput:
    """Tests for audio input creation."""

    @patch("main.MicrophoneInput")
    def test_create_microphone_input(self, mock_mic):
        """Test creating microphone input."""
        mock_mic.return_value = MagicMock()
        create_audio_input("microphone")
        mock_mic.assert_called_once()

    def test_create_stream_input(self):
        """Test creating stream input."""
        from main import StreamInput

        with patch.object(StreamInput, "__init__", return_value=None):
            create_audio_input("stream")

    def test_invalid_source_raises(self):
        """Test that invalid input source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown input source"):
            create_audio_input("invalid")
