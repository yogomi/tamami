"""Main module for the real-time voice translation system.

This module provides the entry point for the audio input to speech recognition pipeline.
"""

import argparse
import signal

from audio.input import AudioInputStream, MicrophoneInput, StreamInput
from speech.recognition import WhisperRecognizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Real-time voice translation system - Audio to text pipeline",
    )
    parser.add_argument(
        "--input-source",
        type=str,
        choices=["microphone", "stream"],
        default="microphone",
        help="Input source for audio (default: microphone)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model name (default: base)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code for transcription (default: auto-detect)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=5.0,
        help="Duration of audio chunks in seconds (default: 5)",
    )
    return parser.parse_args()


def create_audio_input(input_source: str) -> AudioInputStream:
    """Create an audio input stream based on the specified source.

    Args:
        input_source: Type of input source ("microphone" or "stream").

    Returns:
        An AudioInputStream instance.

    Raises:
        ValueError: If input_source is not recognized.
    """
    if input_source == "microphone":
        return MicrophoneInput()
    elif input_source == "stream":
        # For stream input, create a dummy source that returns empty data
        # This is a placeholder for future network-based input
        def dummy_source() -> bytes:
            return b""

        return StreamInput(dummy_source)
    else:
        raise ValueError(f"Unknown input source: {input_source}")


def main() -> None:
    """Execute the audio input to speech recognition pipeline.

    This function runs the main processing loop:
    1. Parses command-line arguments
    2. Initializes audio input source
    3. Initializes Whisper recognizer
    4. Continuously reads audio chunks and transcribes them
    5. Outputs recognized text with language prefix

    Command-line arguments:
        --input-source: Input source ("microphone" or "stream")
        --model: Whisper model name (default: "base")
        --language: Language code (default: None=auto-detect)
        --chunk-duration: Audio chunk duration in seconds (default: 5)

    Examples:
        $ pipenv run start -- --input-source microphone --model base
        $ pipenv run start -- --input-source microphone --language ja
    """
    args = parse_args()

    # Setup signal handler for graceful shutdown
    running = True

    def signal_handler(sig: int, frame: object) -> None:
        nonlocal running
        print("\nShutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Initialize components
    print(f"Initializing audio input ({args.input_source})...")
    audio_input = create_audio_input(args.input_source)

    print(f"Loading Whisper model ({args.model})...")
    recognizer = WhisperRecognizer(model_name=args.model)

    # Calculate chunk size based on sample rate and duration
    sample_rate = audio_input.get_sample_rate()
    chunk_size = int(sample_rate * args.chunk_duration)

    print("Ready. Speak into the microphone (Ctrl+C to exit)...")
    print("-" * 50)

    try:
        while running:
            # Read audio chunk
            audio_data = audio_input.read_chunk(chunk_size)

            # Skip if audio is too quiet (silence detection)
            if audio_data.max() < 0.01:
                continue

            # Transcribe
            result = recognizer.transcribe(audio_data, language=args.language)

            # Output result if there's text
            text = result["text"].strip()
            if text:
                language = result["language"]
                print(f"[{language}] {text}")

    except KeyboardInterrupt:
        pass
    finally:
        audio_input.close()
        print("Closed.")


if __name__ == "__main__":
    main()
