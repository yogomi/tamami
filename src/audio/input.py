"""Audio input module for capturing audio from various sources.

This module provides abstract and concrete implementations for audio input,
including microphone input and stream-based input.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import pyaudio


class AudioInputStream(ABC):
    """Abstract base class for audio input streams.

    This class defines the interface for audio input sources in the
    real-time voice translation system.

    Examples:
        >>> class CustomInput(AudioInputStream):
        ...     def read_chunk(self, chunk_size: int) -> np.ndarray:
        ...         return np.zeros(chunk_size, dtype=np.float32)
        ...     def get_sample_rate(self) -> int:
        ...         return 16000
        ...     def close(self) -> None:
        ...         pass
    """

    @abstractmethod
    def read_chunk(self, chunk_size: int) -> np.ndarray:
        """Read a chunk of audio data from the input stream.

        Args:
            chunk_size: Number of samples to read.

        Returns:
            Audio data as a numpy array with float32 dtype.

        Raises:
            IOError: If reading from the audio source fails.
        """
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Get the sample rate of the audio stream.

        Returns:
            Sample rate in Hz.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the audio stream and release resources.

        This method should be called when the audio input is no longer needed
        to properly release system resources.
        """
        pass


class MicrophoneInput(AudioInputStream):
    """Audio input from microphone using PyAudio.

    This class captures audio from the system's default microphone device
    at 16000Hz sample rate in mono format, which is optimized for Whisper.

    Args:
        device_index: Optional device index for the microphone.
            If None, uses the default input device.

    Examples:
        >>> mic = MicrophoneInput()
        >>> audio_data = mic.read_chunk(16000)  # Read 1 second of audio
        >>> sample_rate = mic.get_sample_rate()
        >>> mic.close()
    """

    SAMPLE_RATE = 16000
    CHANNELS = 1
    FORMAT = pyaudio.paFloat32
    CHUNK_SIZE = 1024

    def __init__(self, device_index: Optional[int] = None) -> None:
        """Initialize microphone input.

        Args:
            device_index: Optional device index for the microphone.
                If None, uses the default input device.

        Raises:
            IOError: If the microphone device cannot be opened.
        """
        self._device_index = device_index
        self._audio = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        self._buffer = np.array([], dtype=np.float32)
        self._open_stream()

    def _open_stream(self) -> None:
        """Open the audio stream from the microphone.

        Raises:
            IOError: If the stream cannot be opened.
        """
        try:
            self._stream = self._audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                input_device_index=self._device_index,
                frames_per_buffer=self.CHUNK_SIZE,
            )
        except Exception as e:
            raise IOError(f"Failed to open microphone: {e}") from e

    def read_chunk(self, chunk_size: int) -> np.ndarray:
        """Read a chunk of audio data from the microphone.

        Args:
            chunk_size: Number of samples to read.

        Returns:
            Audio data as a numpy array with float32 dtype.

        Raises:
            IOError: If reading from the microphone fails.

        Examples:
            >>> mic = MicrophoneInput()
            >>> audio = mic.read_chunk(16000)  # Read 1 second at 16kHz
            >>> print(audio.shape)
            (16000,)
            >>> mic.close()
        """
        if self._stream is None:
            raise IOError("Microphone stream is not open.")

        # Read from stream until we have enough samples
        while len(self._buffer) < chunk_size:
            try:
                data = self._stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                self._buffer = np.concatenate([self._buffer, audio_chunk])
            except Exception as e:
                raise IOError(f"Failed to read from microphone: {e}") from e

        # Extract requested chunk and keep remainder in buffer
        result = self._buffer[:chunk_size]
        self._buffer = self._buffer[chunk_size:]
        return result

    def get_sample_rate(self) -> int:
        """Get the sample rate of the microphone input.

        Returns:
            Sample rate in Hz (16000).
        """
        return self.SAMPLE_RATE

    def close(self) -> None:
        """Close the microphone stream and release resources.

        Examples:
            >>> mic = MicrophoneInput()
            >>> mic.close()
        """
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._audio is not None:
            self._audio.terminate()


class StreamInput(AudioInputStream):
    """Audio input from a stream source (socket, file, etc.).

    This class handles audio input from stream-based sources such as
    network sockets or file streams. It is designed for future integration
    with network-based audio transmission.

    Args:
        data_source: Callable that returns audio data as bytes when called.
        sample_rate: Sample rate of the audio data in Hz. Defaults to 16000.

    Examples:
        >>> import io
        >>> buffer = io.BytesIO(np.zeros(16000, dtype=np.float32).tobytes())
        >>> def read_fn():
        ...     return buffer.read(4096)
        >>> stream = StreamInput(read_fn, sample_rate=16000)
        >>> audio = stream.read_chunk(1000)
        >>> stream.close()
    """

    def __init__(
        self,
        data_source: Callable[[], bytes],
        sample_rate: int = 16000,
    ) -> None:
        """Initialize stream input.

        Args:
            data_source: Callable that returns audio data as bytes when called.
                Should return empty bytes when no more data is available.
            sample_rate: Sample rate of the audio data in Hz. Defaults to 16000.
        """
        self._data_source = data_source
        self._sample_rate = sample_rate
        self._buffer = np.array([], dtype=np.float32)
        self._closed = False

    def read_chunk(self, chunk_size: int) -> np.ndarray:
        """Read a chunk of audio data from the stream.

        Args:
            chunk_size: Number of samples to read.

        Returns:
            Audio data as a numpy array with float32 dtype.
            If not enough data is available, returns available data
            padded with zeros.

        Raises:
            IOError: If the stream is closed or reading fails.

        Examples:
            >>> def data_fn():
            ...     return np.ones(1000, dtype=np.float32).tobytes()
            >>> stream = StreamInput(data_fn)
            >>> audio = stream.read_chunk(500)
            >>> print(len(audio))
            500
            >>> stream.close()
        """
        if self._closed:
            raise IOError("Stream is closed.")

        # Try to fill buffer with enough data
        while len(self._buffer) < chunk_size:
            try:
                data = self._data_source()
                if not data:
                    # No more data available, pad with zeros
                    if len(self._buffer) == 0:
                        return np.zeros(chunk_size, dtype=np.float32)
                    result = np.zeros(chunk_size, dtype=np.float32)
                    result[: len(self._buffer)] = self._buffer
                    self._buffer = np.array([], dtype=np.float32)
                    return result
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                self._buffer = np.concatenate([self._buffer, audio_chunk])
            except Exception as e:
                raise IOError(f"Failed to read from stream: {e}") from e

        # Extract requested chunk and keep remainder in buffer
        result = self._buffer[:chunk_size]
        self._buffer = self._buffer[chunk_size:]
        return result

    def get_sample_rate(self) -> int:
        """Get the sample rate of the stream input.

        Returns:
            Sample rate in Hz.
        """
        return self._sample_rate

    def close(self) -> None:
        """Close the stream input.

        Examples:
            >>> def data_fn():
            ...     return b''
            >>> stream = StreamInput(data_fn)
            >>> stream.close()
        """
        self._closed = True
        self._buffer = np.array([], dtype=np.float32)
