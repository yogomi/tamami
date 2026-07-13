"""server.app モジュールのテスト.

aiohttp.test_utils.TestClient/TestServerを使い、実際のWebSocket接続を通して
ConnectionHandlerの振る舞いを検証する。WebRTC(webrtc_offer)の実確立は重いため
ここでは扱わず、session_start/ping/session_end・認識器イベントの配送・
致命的エラー時の接続クローズに絞る。recognizer_factoryにはStreamingRecognizerの
インターフェースだけを満たすスタブを差し替える。
"""

import asyncio

import numpy as np
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from src.server import protocol
from src.server.app import create_app
from src.speech.fake import FakeStreamingRecognizer
from src.speech.streaming import AsrEvent, RecognizerOverloadedError


class _StubRecognizer:
    """StreamingRecognizerのインターフェースだけを満たすテスト用スタブ."""

    def __init__(self) -> None:
        self.fed: list = []
        self.flushed = False
        self.closed = False
        self._queue: "asyncio.Queue[object]" = asyncio.Queue()

    def feed(self, pcm) -> None:
        self.fed.append(pcm)

    async def events(self):
        while True:
            item = await self._queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def push_event(self, event: AsrEvent) -> None:
        """テストからevents()へ配送するイベントを積む."""
        self._queue.put_nowait(event)

    def push_error(self, exc: BaseException) -> None:
        """テストからevents()へ配送する例外を積む."""
        self._queue.put_nowait(exc)

    async def flush(self) -> None:
        self.flushed = True

    async def close(self) -> None:
        self.closed = True
        self._queue.put_nowait(None)


def _make_factory():
    """呼び出しごとに_StubRecognizerを生成し、生成物を記録するfactoryを返す.

    Returns:
        (recognizer_factory, created) のタプル。createdには生成された
        スタブが順に追加される。
    """
    created: list[_StubRecognizer] = []

    def factory() -> _StubRecognizer:
        stub = _StubRecognizer()
        created.append(stub)
        return stub

    return factory, created


def _session_start_message() -> dict:
    """テスト用のsession_startメッセージを組み立てる."""
    return {
        "type": "session_start",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "client_ts_ms": 1751400000000,
    }


async def _wait_until(condition, timeout: float = 2.0) -> None:
    """条件が真になるまでポーリングする（テストの同期待ち用）.

    Args:
        condition: 引数なしで真偽を返す呼び出し可能オブジェクト。
        timeout: 最大待機秒数。

    Raises:
        AssertionError: timeoutまでに条件が真にならなかった場合。
    """
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


class TestSessionStart:
    """session_start処理のテスト."""

    def test_session_ready_returned(self):
        async def scenario():
            factory, created = _make_factory()
            app = create_app(factory)
            async with TestClient(TestServer(app)) as client:
                async with client.ws_connect("/ws") as ws:
                    await ws.send_json(_session_start_message())
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "session_ready"
                    assert msg["protocol_version"] == protocol.PROTOCOL_VERSION
                    assert isinstance(msg["session_id"], str)
            assert len(created) == 1

        asyncio.run(scenario())

    def test_duplicate_session_start_is_fatal_error(self):
        async def scenario():
            factory, _ = _make_factory()
            app = create_app(factory)
            async with TestClient(TestServer(app)) as client:
                async with client.ws_connect("/ws") as ws:
                    await ws.send_json(_session_start_message())
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await ws.send_json(_session_start_message())
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "error"
                    assert msg["code"] == "invalid_config"
                    assert msg["fatal"] is True

        asyncio.run(scenario())

    def test_unsupported_version_is_fatal_error(self):
        async def scenario():
            factory, _ = _make_factory()
            app = create_app(factory)
            async with TestClient(TestServer(app)) as client:
                async with client.ws_connect("/ws") as ws:
                    await ws.send_json(
                        {
                            "type": "session_start",
                            "protocol_version": 999,
                            "client_ts_ms": 0,
                        }
                    )
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "error"
                    assert msg["code"] == "unsupported_version"
                    assert msg["fatal"] is True

        asyncio.run(scenario())


class TestPing:
    """pingへの応答のテスト."""

    def test_ping_returns_pong(self):
        async def scenario():
            factory, _ = _make_factory()
            app = create_app(factory)
            async with TestClient(TestServer(app)) as client:
                async with client.ws_connect("/ws") as ws:
                    await ws.send_json({"type": "ping", "client_ts_ms": 42})
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "pong"
                    assert msg["client_ts_ms"] == 42

        asyncio.run(scenario())


class TestAsrEventDelivery:
    """認識器からのAsrEventがasrメッセージとして配信されることのテスト."""

    def test_asr_event_is_delivered(self):
        async def scenario():
            factory, created = _make_factory()
            app = create_app(factory)
            async with TestClient(TestServer(app)) as client:
                async with client.ws_connect("/ws") as ws:
                    await ws.send_json(_session_start_message())
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)  # session_ready

                    await _wait_until(lambda: len(created) == 1)
                    stub = created[0]
                    event = AsrEvent(
                        segment_id=3,
                        text="こんにちは",
                        lang="ja",
                        is_final=True,
                        ts_audio_start=1.0,
                        ts_audio_end=2.5,
                    )
                    stub.push_event(event)

                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "asr"
                    assert msg["segment_id"] == 3
                    assert msg["text"] == "こんにちは"
                    assert msg["lang"] == "ja"
                    assert msg["is_final"] is True
                    assert msg["ts_audio_start"] == 1.0
                    assert msg["ts_audio_end"] == 2.5

        asyncio.run(scenario())


class TestFatalRecognizerError:
    """認識器の致命的エラーがerrorメッセージ配信 + 接続クローズになることのテスト."""

    def test_overloaded_closes_connection(self):
        async def scenario():
            factory, created = _make_factory()
            app = create_app(factory)
            async with TestClient(TestServer(app)) as client:
                async with client.ws_connect("/ws") as ws:
                    await ws.send_json(_session_start_message())
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)  # session_ready

                    await _wait_until(lambda: len(created) == 1)
                    stub = created[0]
                    stub.push_error(RecognizerOverloadedError("too slow"))

                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "error"
                    assert msg["code"] == "overloaded"
                    assert msg["fatal"] is True

                    # サーバー側がwsをcloseするため、後続の受信はclose系になる
                    closing_msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    assert closing_msg.type in (
                        WSMsgType.CLOSE,
                        WSMsgType.CLOSING,
                        WSMsgType.CLOSED,
                    )
                    await _wait_until(lambda: stub.closed is True)

        asyncio.run(scenario())


class TestSessionEnd:
    """session_end処理のテスト."""

    def test_session_end_flushes_and_closes(self):
        async def scenario():
            factory, created = _make_factory()
            app = create_app(factory)
            async with TestClient(TestServer(app)) as client:
                async with client.ws_connect("/ws") as ws:
                    await ws.send_json(_session_start_message())
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)  # session_ready

                    await _wait_until(lambda: len(created) == 1)
                    stub = created[0]

                    await ws.send_json({"type": "session_end"})

                    await _wait_until(lambda: stub.flushed is True)
                    await _wait_until(lambda: stub.closed is True)

        asyncio.run(scenario())


class TestAsrLifecycleIntegration:
    """FakeStreamingRecognizerを実際に通した結合テスト.

    レビュー指摘: 従来の_StubRecognizerはevents()へイベントを積む主体が
    テストコード自身だったため、「session_end時に確定asrが接続クローズ前に
    届く」というPROTOCOL.mdの契約（recognizer.flush()のドレイン契約 +
    StreamingSession.close()がdispatch_taskの自然終了を待つこと）を実際には
    検証できていなかった。ここでは本物のFakeStreamingRecognizerを
    recognizer_factoryに差し込み、音声feed→session_endの経路を通しで確認する。
    WebRTCトラックの受信自体（_consume）はtest_session.pyで別途検証済みのため、
    ここではrecognizer.feed()を直接呼んで音声到着を模する。
    """

    def test_final_asr_delivered_before_close_on_session_end(self):
        async def scenario():
            created: list[FakeStreamingRecognizer] = []

            def factory() -> FakeStreamingRecognizer:
                recognizer = FakeStreamingRecognizer(min_silence_sec=0.3)
                created.append(recognizer)
                return recognizer

            app = create_app(factory)
            async with TestClient(TestServer(app)) as client:
                async with client.ws_connect("/ws") as ws:
                    await ws.send_json(_session_start_message())
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)  # session_ready

                    await _wait_until(lambda: len(created) == 1)
                    recognizer = created[0]

                    # chunk_samples(0.5秒)ちょうどの「発話」を模したPCMをfeedし、
                    # 暫定asr(is_final=False)を1件発生させる
                    n = 8000
                    loud = (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(n) / 16000)).astype(
                        np.float32
                    )
                    recognizer.feed(loud)

                    interim = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert interim["type"] == "asr"
                    assert interim["is_final"] is False

                    # session_end -> flush()で進行中セグメントが強制確定され、
                    # StreamingSession.close()がdispatch_taskの自然終了を待つため、
                    # 確定asrは接続クローズ前にこの接続へ届くはず
                    await ws.send_json({"type": "session_end"})

                    final_msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert final_msg["type"] == "asr"
                    assert final_msg["is_final"] is True
                    assert final_msg["segment_id"] == interim["segment_id"]

                    # 確定asrの後、接続はクローズ系メッセージに向かう
                    closing = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    assert closing.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED)

        asyncio.run(scenario())
