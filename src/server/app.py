"""シグナリング + 結果配信のWebSocketサーバー.

PROTOCOL.mdに定義された制御チャネル（`/ws`）を提供する。接続ごとの流れ:
session_start → session_ready → webrtc_offer → webrtc_answer →
音声受信・ストリーミング認識（asrとして配信）→ session_end。
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Optional

from aiohttp import WSMsgType, web

from src.server import protocol
from src.server.session import StreamingSession
from src.speech.streaming import AsrEvent, StreamingRecognizer

logger = logging.getLogger(__name__)

# セッションごとに新しいStreamingRecognizerを生成するファクトリの型
RecognizerFactory = Callable[[], StreamingRecognizer]

# aiohttp.web.Applicationにrecognizer_factoryを持たせる際のキー
# （aiohttpの推奨に従い、プレーンな文字列キーではなくAppKeyを使う）
_RECOGNIZER_FACTORY_KEY: web.AppKey[RecognizerFactory] = web.AppKey(
    "recognizer_factory", RecognizerFactory
)


class ConnectionHandler:
    """1つのWebSocket接続のプロトコル状態を管理する.

    Args:
        ws: 確立済みのWebSocketレスポンス。
        recognizer_factory: セッションごとに新しいStreamingRecognizerを
            生成するファクトリ。

    Attributes:
        なし（すべて非公開状態として保持する）。
    """

    def __init__(self, ws: web.WebSocketResponse, recognizer_factory: RecognizerFactory) -> None:
        """ハンドラを初期化する.

        Args:
            ws: 確立済みのWebSocketレスポンス。
            recognizer_factory: セッションごとに新しいStreamingRecognizerを
                生成するファクトリ。
        """
        self._ws = ws
        self._recognizer_factory = recognizer_factory
        self._send_lock = asyncio.Lock()
        self._session_id: Optional[str] = None
        self._session: Optional[StreamingSession] = None

    async def run(self) -> None:
        """メッセージループを実行する.

        受信メッセージをディスパッチし、プロトコル違反時はerrorを送って
        必要に応じて接続を閉じる。終了時にセッション資源を解放する。
        """
        try:
            async for message in self._ws:
                if message.type == WSMsgType.TEXT:
                    should_continue = await self._handle_text(message.data)
                    if not should_continue:
                        break
                elif message.type == WSMsgType.BINARY:
                    # バージョン 1 では上りバイナリフレームは未定義（音声はWebRTC側）
                    logger.warning("[%s] unexpected binary frame ignored", self._session_id)
                elif message.type == WSMsgType.ERROR:
                    logger.warning(
                        "[%s] websocket error: %s",
                        self._session_id,
                        self._ws.exception(),
                    )
                    break
        finally:
            if self._session is not None:
                await self._session.close()
            logger.info("[%s] connection closed", self._session_id)

    async def _handle_text(self, raw: str) -> bool:
        """テキストフレーム1件を処理する.

        Args:
            raw: 受信したJSON文字列。

        Returns:
            接続を継続する場合True、閉じる場合False。
        """
        try:
            message = protocol.parse_client_message(raw)
            return await self._dispatch(message)
        except protocol.ProtocolError as e:
            logger.warning("[%s] protocol error: %s (%s)", self._session_id, e.message, e.code)
            await self._send(protocol.make_error(e.code, e.message, e.fatal))
            return not e.fatal
        except Exception as e:
            logger.exception("[%s] internal error", self._session_id)
            await self._send(protocol.make_error("internal_error", str(e), True))
            return False

    async def _dispatch(self, message: dict[str, Any]) -> bool:
        """メッセージ種別ごとの処理に振り分ける.

        Args:
            message: 解析済みのクライアントメッセージ。

        Returns:
            接続を継続する場合True、閉じる場合False。

        Raises:
            ProtocolError: プロトコル順序違反・内容不正の場合。
        """
        message_type = message["type"]
        if message_type == "session_start":
            await self._on_session_start(message)
        elif message_type == "webrtc_offer":
            await self._on_webrtc_offer(message)
        elif message_type == "ping":
            client_ts_ms = message.get("client_ts_ms")
            if not isinstance(client_ts_ms, int):
                raise protocol.ProtocolError("invalid_config", "ping requires client_ts_ms", False)
            await self._send(protocol.make_pong(client_ts_ms))
        elif message_type == "session_end":
            await self._on_session_end()
            return False
        else:
            # 前方互換のため未知の種別は無視する
            logger.warning("[%s] unknown message type ignored: %s", self._session_id, message_type)
        return True

    async def _on_session_start(self, message: dict[str, Any]) -> None:
        """session_startを処理し、session_readyを返す.

        Args:
            message: session_startメッセージ。

        Raises:
            ProtocolError: 検証失敗、または二重送信の場合。
        """
        if self._session_id is not None:
            raise protocol.ProtocolError("invalid_config", "session already started")
        protocol.validate_session_start(message)
        self._session_id = uuid.uuid4().hex[:8]
        recognizer = self._recognizer_factory()
        self._session = StreamingSession(
            self._session_id, recognizer, self._on_asr_event, self._on_fatal
        )
        await self._send(protocol.make_session_ready(self._session_id))
        logger.info("[%s] session started", self._session_id)

    async def _on_webrtc_offer(self, message: dict[str, Any]) -> None:
        """webrtc_offerを処理し、webrtc_answerを返す.

        Args:
            message: webrtc_offerメッセージ。

        Raises:
            ProtocolError: session_start前の受信、SDP不正、確立失敗の場合。
        """
        if self._session is None:
            raise protocol.ProtocolError("invalid_config", "webrtc_offer before session_start")
        sdp = message.get("sdp")
        if not isinstance(sdp, str) or not sdp:
            raise protocol.ProtocolError("webrtc_failure", "webrtc_offer requires sdp")
        try:
            answer_sdp = await self._session.handle_offer(sdp)
        except Exception as e:
            raise protocol.ProtocolError("webrtc_failure", f"failed to establish: {e}") from e
        await self._send(protocol.make_webrtc_answer(answer_sdp))
        logger.info("[%s] webrtc answer sent", self._session_id)

    async def _on_session_end(self) -> None:
        """session_endを処理する.

        処理中の音声をフラッシュし、確定結果を出し切るまで待つ。呼び出し後、
        run()のループが接続を閉じる。
        """
        logger.info("[%s] session end requested", self._session_id)
        if self._session is not None:
            await self._session.flush()

    async def _on_asr_event(self, event: AsrEvent) -> None:
        """認識器からのAsrEventをasrメッセージとして配信する.

        Args:
            event: 認識結果イベント。
        """
        await self._send(
            protocol.make_asr(
                event.segment_id,
                event.text,
                event.lang,
                event.is_final,
                event.ts_audio_start,
                event.ts_audio_end,
            )
        )

    async def _on_fatal(self, code: str, message: str) -> None:
        """認識器からの致命的エラーをerrorメッセージとして配信し、接続を閉じる.

        Args:
            code: PROTOCOL.mdに定義されたエラーコード（"overloaded"等）。
            message: エラー内容。
        """
        logger.error("[%s] fatal recognizer error: %s (%s)", self._session_id, message, code)
        await self._send(protocol.make_error(code, message, True))
        await self._ws.close()

    async def _send(self, message: dict[str, Any]) -> None:
        """JSONメッセージを送信する（複数タスクからの送信を直列化する）.

        Args:
            message: 送信するメッセージ。
        """
        if self._ws.closed:
            return
        async with self._send_lock:
            await self._ws.send_str(json.dumps(message, ensure_ascii=False))


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """`/ws`エンドポイントのハンドラ.

    Args:
        request: aiohttpのリクエスト。

    Returns:
        クローズ済みのWebSocketレスポンス。
    """
    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)
    recognizer_factory = request.app[_RECOGNIZER_FACTORY_KEY]
    handler = ConnectionHandler(ws, recognizer_factory)
    await handler.run()
    await ws.close()
    return ws


def create_app(recognizer_factory: RecognizerFactory) -> web.Application:
    """WebSocketサーバーのaiohttpアプリケーションを生成する.

    Args:
        recognizer_factory: セッションごとに新しいStreamingRecognizerを
            生成するファクトリ。

    Returns:
        `/ws`ルートを持つaiohttpアプリケーション。
    """
    app = web.Application()
    app[_RECOGNIZER_FACTORY_KEY] = recognizer_factory
    app.router.add_get("/ws", websocket_handler)
    return app
