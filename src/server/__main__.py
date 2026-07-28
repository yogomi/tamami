"""ストリーミング翻訳サーバーのエントリポイント.

Examples:
    $ uv run python -m src.server
    $ uv run python -m src.server --host 0.0.0.0 --port 8765 --asr fake
    $ uv run python -m src.server --asr nemotron --chunk-ms 560
"""

import argparse
import logging
import sys

from aiohttp import web

from src.server.app import RecognizerFactory, create_app
from src.speech.fake import FakeStreamingRecognizer

logger = logging.getLogger(__name__)

# NemotronStreamingRecognizerが対応するチャンク長（ミリ秒）。
# Transformersのlookahead段階に対応する（src/speech/nemotron.pyのCHUNK_MS_TO_LOOKAHEAD参照）。
CHUNK_MS_CHOICES = [80, 320, 560, 1120]


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する.

    Returns:
        解析済みの引数。
    """
    parser = argparse.ArgumentParser(description="tamami streaming translation server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=8765, help="listen port (default: 8765)")
    parser.add_argument(
        "--asr",
        choices=["fake", "nemotron"],
        default="fake",
        help="使用するASR実装（デフォルト: fake。GPUがなくても動作する）",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        choices=CHUNK_MS_CHOICES,
        default=560,
        help="nemotron使用時のチャンクサイズ（ミリ秒）。fake使用時は無視される",
    )
    return parser.parse_args()


def build_recognizer_factory(args: argparse.Namespace) -> RecognizerFactory:
    """--asrオプションに応じたrecognizer_factoryを組み立てる.

    Args:
        args: parse_args()で解析済みの引数。

    Returns:
        呼び出すたびに新しいStreamingRecognizerを生成する関数。

    副作用:
        --asr nemotron の場合、NemotronStreamingRecognizer.load_model()を
        呼んでプロセス内で1回だけモデルをロードする（起動時にブロッキングで
        実行し、失敗した場合はプロセスを終了する）。
    """
    if args.asr == "fake":
        return FakeStreamingRecognizer

    # --asr nemotron: NeMoに依存するためモジュールのimportをここまで遅延させる
    # （Mac等NeMo未インストール環境でも--asr fakeなら起動できるようにするため）。
    from src.speech.nemotron import NemotronStreamingRecognizer

    try:
        NemotronStreamingRecognizer.load_model(args.chunk_ms)
    except Exception:
        logger.exception("failed to load Nemotron ASR model")
        sys.exit(1)
    return lambda: NemotronStreamingRecognizer(chunk_ms=args.chunk_ms)


def main() -> None:
    """サーバーを起動する."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    recognizer_factory = build_recognizer_factory(args)
    web.run_app(create_app(recognizer_factory), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
