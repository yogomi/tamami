"""server.__main__ モジュールのテスト."""

from unittest.mock import patch

import pytest

from src.server.__main__ import build_recognizer_factory, parse_args
from src.speech.fake import FakeStreamingRecognizer


class TestParseArgs:
    """コマンドライン引数解析のテスト."""

    def test_default_args(self):
        with patch("sys.argv", ["__main__.py"]):
            args = parse_args()
            assert args.host == "0.0.0.0"
            assert args.port == 8765
            assert args.asr == "fake"
            assert args.chunk_ms == 560

    def test_custom_args(self):
        with patch(
            "sys.argv",
            [
                "__main__.py",
                "--host",
                "127.0.0.1",
                "--port",
                "9000",
                "--asr",
                "nemotron",
                "--chunk-ms",
                "320",
            ],
        ):
            args = parse_args()
            assert args.host == "127.0.0.1"
            assert args.port == 9000
            assert args.asr == "nemotron"
            assert args.chunk_ms == 320

    def test_invalid_asr_choice_raises(self):
        with patch("sys.argv", ["__main__.py", "--asr", "whisper"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_invalid_chunk_ms_choice_raises(self):
        with patch("sys.argv", ["__main__.py", "--chunk-ms", "999"]):
            with pytest.raises(SystemExit):
                parse_args()


class TestBuildRecognizerFactory:
    """recognizer_factory組み立てのテスト."""

    def test_fake_returns_fake_streaming_recognizer_factory(self):
        with patch("sys.argv", ["__main__.py", "--asr", "fake"]):
            args = parse_args()
        factory = build_recognizer_factory(args)
        assert factory is FakeStreamingRecognizer

    def test_nemotron_loads_model_and_exits_on_failure(self):
        with patch("sys.argv", ["__main__.py", "--asr", "nemotron", "--chunk-ms", "320"]):
            args = parse_args()
        with patch(
            "src.speech.nemotron.NemotronStreamingRecognizer.load_model",
            side_effect=RuntimeError("no GPU"),
        ):
            with pytest.raises(SystemExit):
                build_recognizer_factory(args)
