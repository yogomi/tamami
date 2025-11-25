"""Tests for the audio input module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.audio.input import AudioInputStream, MicrophoneInput, StreamInput


class TestAudioInputStream:
    """Tests for the AudioInputStream abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that AudioInputStream cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AudioInputStream()


class TestStreamInput:
    """Tests for the StreamInput class."""

    def test_init(self):
        """Test StreamInput initialization."""

        def data_fn():
            return b""

        stream = StreamInput(data_fn, sample_rate=16000)
        assert stream.get_sample_rate() == 16000
        stream.close()

    def test_read_chunk_with_data(self):
        """Test reading a chunk from stream with available data."""
        test_data = np.ones(1000, dtype=np.float32)
        data_iter = iter([test_data.tobytes(), b""])

        def data_fn():
            return next(data_iter, b"")

        stream = StreamInput(data_fn, sample_rate=16000)
        result = stream.read_chunk(500)

        assert len(result) == 500
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, np.ones(500, dtype=np.float32))
        stream.close()

    def test_read_chunk_no_data_returns_zeros(self):
        """Test that reading with no data returns zeros."""

        def data_fn():
            return b""

        stream = StreamInput(data_fn)
        result = stream.read_chunk(100)

        assert len(result) == 100
        np.testing.assert_array_equal(result, np.zeros(100, dtype=np.float32))
        stream.close()

    def test_read_chunk_after_close_raises(self):
        """Test that reading after close raises IOError."""

        def data_fn():
            return b""

        stream = StreamInput(data_fn)
        stream.close()

        with pytest.raises(IOError):
            stream.read_chunk(100)

    def test_get_sample_rate(self):
        """Test getting sample rate."""

        def data_fn():
            return b""

        stream = StreamInput(data_fn, sample_rate=44100)
        assert stream.get_sample_rate() == 44100
        stream.close()


class TestMicrophoneInput:
    """Tests for the MicrophoneInput class."""

    @patch("src.audio.input.pyaudio.PyAudio")
    def test_init(self, mock_pyaudio):
        """Test MicrophoneInput initialization."""
        mock_audio = MagicMock()
        mock_pyaudio.return_value = mock_audio

        mic = MicrophoneInput()

        mock_audio.open.assert_called_once()
        assert mic.get_sample_rate() == 16000
        mic.close()

    @patch("src.audio.input.pyaudio.PyAudio")
    def test_read_chunk(self, mock_pyaudio):
        """Test reading a chunk from microphone."""
        mock_audio = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio.return_value = mock_audio
        mock_audio.open.return_value = mock_stream

        # Create test audio data
        test_data = np.ones(1024, dtype=np.float32)
        mock_stream.read.return_value = test_data.tobytes()

        mic = MicrophoneInput()
        result = mic.read_chunk(500)

        assert len(result) == 500
        assert result.dtype == np.float32
        mic.close()

    @patch("src.audio.input.pyaudio.PyAudio")
    def test_close(self, mock_pyaudio):
        """Test closing microphone."""
        mock_audio = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio.return_value = mock_audio
        mock_audio.open.return_value = mock_stream

        mic = MicrophoneInput()
        mic.close()

        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_audio.terminate.assert_called_once()

    @patch("src.audio.input.pyaudio.PyAudio")
    def test_init_failure(self, mock_pyaudio):
        """Test MicrophoneInput handles initialization failure."""
        mock_audio = MagicMock()
        mock_pyaudio.return_value = mock_audio
        mock_audio.open.side_effect = Exception("No microphone available")

        with pytest.raises(IOError, match="Failed to open microphone"):
            MicrophoneInput()
