"""ストリーミング翻訳サーバーのエントリポイント.

Examples:
    $ pipenv run serve
    $ python -m src.server --host 0.0.0.0 --port 8765
"""

import argparse
import logging

from aiohttp import web

from src.server.app import create_app


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する.

    Returns:
        解析済みの引数。
    """
    parser = argparse.ArgumentParser(description="tamami streaming translation server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="bind address")
    parser.add_argument(
        "--port", type=int, default=8765, help="listen port (default: 8765)"
    )
    return parser.parse_args()


def main() -> None:
    """サーバーを起動する."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    web.run_app(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
