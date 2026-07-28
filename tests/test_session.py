"""server.session モジュールのテスト.

RTCPeerConnectionの実確立（ICE/DTLS）は重いため、_consume()には直接
フェイクの音声トラックを渡して単体テストする。認識器はStreamingRecognizerの
インターフェース（feed/events/flush/close）だけを満たすスタブに差し替える。
"""

import asyncio
from typing import AsyncIterator, Optional, Union

import av
import numpy as np
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

from src.server.session import StreamingSession
from src.speech.streaming import AsrEvent, RecognizerOverloadedError, StreamingRecognizer


def _make_frame(num_samples: int, value: int, rate: int = 48000) -> av.AudioFrame:
    """テスト用のs16モノラルAudioFrameを生成する.

    Args:
        num_samples: サンプル数。
        value: 全サンプルに設定するint16値。
        rate: サンプルレート（Hz）。

    Returns:
        指定条件のav.AudioFrame。
    """
    data = np.full((1, num_samples), value, dtype=np.int16)
    frame = av.AudioFrame.from_ndarray(data, format="s16", layout="mono")
    frame.sample_rate = rate
    frame.pts = 0
    return frame


class _FakeAudioTrack(MediaStreamTrack):
    """recv()が固定フレーム列を返した後MediaStreamErrorで終端するフェイクトラック."""

    kind = "audio"

    def __init__(self, frames: list[av.AudioFrame]) -> None:
        super().__init__()
        self._frames = list(frames)

    async def recv(self) -> av.AudioFrame:
        if not self._frames:
            raise MediaStreamError()
        return self._frames.pop(0)


class _StubRecognizer(StreamingRecognizer):
    """StreamingRecognizerの公開APIだけを差し替えたテスト用スタブ.

    実際のワーカースレッドは持たず、テストコードからasyncio.Queueへ直接
    イベント・例外を積んでevents()経由で配送できるようにする。

    基底の__init__はワーカースレッドを起動し実行中のイベントループを要求するため、
    意図的に呼ばない（公開API（feed/events/flush/close）をすべて上書きしており、
    基底の内部状態には触れない）。継承しているのは、StreamingRecognizerを要求する
    引数へ型として渡せるようにするため。
    """

    def __init__(self) -> None:
        self.fed: list[np.ndarray] = []
        self.flushed = False
        self.closed = False
        self._queue: "asyncio.Queue[Optional[Union[AsrEvent, BaseException]]]" = asyncio.Queue()

    def feed(self, pcm: np.ndarray) -> None:
        self.fed.append(pcm)

    async def events(self) -> AsyncIterator[AsrEvent]:
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


async def _noop_on_event(event: AsrEvent) -> None:
    """未使用のon_eventコールバック（テストで結果を検証しない場合用）."""


async def _noop_on_fatal(code: str, message: str) -> None:
    """未使用のon_fatalコールバック（テストで結果を検証しない場合用）."""


class TestConsume:
    """_consume()による正規化・feed・received_secondsのテスト."""

    def test_feed_receives_normalized_float32(self):
        async def scenario():
            stub = _StubRecognizer()
            session = StreamingSession("sid", stub, _noop_on_event, _noop_on_fatal)
            # int16の半分の値 -> float32では約0.5になるはず
            track = _FakeAudioTrack([_make_frame(960, value=16384)])
            await session._consume(track)

            assert stub.fed, "recognizer.feed()が呼ばれているはず"
            pcm = np.concatenate(stub.fed)
            assert pcm.dtype == np.float32
            assert np.all(pcm <= 1.0) and np.all(pcm >= -1.0)
            assert np.allclose(pcm, 16384 / 32768.0, atol=0.02)
            assert session.received_seconds > 0.0
            await session.close()

        asyncio.run(scenario())

    def test_received_seconds_zero_before_any_audio(self):
        async def scenario():
            stub = _StubRecognizer()
            session = StreamingSession("sid", stub, _noop_on_event, _noop_on_fatal)
            assert session.received_seconds == 0.0
            await session.close()

        asyncio.run(scenario())


class TestDispatchEvents:
    """recognizer.events()の配送先振り分けのテスト."""

    def test_asr_event_is_forwarded_to_on_event(self):
        async def scenario():
            stub = _StubRecognizer()
            received: list[AsrEvent] = []
            got = asyncio.Event()

            async def on_event(event: AsrEvent) -> None:
                received.append(event)
                got.set()

            session = StreamingSession("sid", stub, on_event, _noop_on_fatal)
            event = AsrEvent(0, "text", "ja", True, 0.0, 1.0)
            stub.push_event(event)
            await asyncio.wait_for(got.wait(), timeout=2.0)

            assert received == [event]
            await session.close()

        asyncio.run(scenario())

    def test_overloaded_error_maps_to_on_fatal_overloaded(self):
        async def scenario():
            stub = _StubRecognizer()
            fatal_calls: list[tuple[str, str]] = []
            got = asyncio.Event()

            async def on_fatal(code: str, message: str) -> None:
                fatal_calls.append((code, message))
                got.set()

            session = StreamingSession("sid", stub, _noop_on_event, on_fatal)
            stub.push_error(RecognizerOverloadedError("too slow"))
            await asyncio.wait_for(got.wait(), timeout=2.0)

            assert fatal_calls[0][0] == "overloaded"
            await session.close()

        asyncio.run(scenario())

    def test_generic_exception_maps_to_on_fatal_internal_error(self):
        async def scenario():
            stub = _StubRecognizer()
            fatal_calls: list[tuple[str, str]] = []
            got = asyncio.Event()

            async def on_fatal(code: str, message: str) -> None:
                fatal_calls.append((code, message))
                got.set()

            session = StreamingSession("sid", stub, _noop_on_event, on_fatal)
            stub.push_error(RuntimeError("boom"))
            await asyncio.wait_for(got.wait(), timeout=2.0)

            assert fatal_calls[0][0] == "internal_error"
            await session.close()

        asyncio.run(scenario())


class TestFlushAndClose:
    """flush()・close()の委譲・冪等性のテスト."""

    def test_flush_delegates_to_recognizer(self):
        async def scenario():
            stub = _StubRecognizer()
            session = StreamingSession("sid", stub, _noop_on_event, _noop_on_fatal)
            await session.flush()
            assert stub.flushed is True
            await session.close()

        asyncio.run(scenario())

    def test_close_delegates_and_is_idempotent(self):
        async def scenario():
            stub = _StubRecognizer()
            session = StreamingSession("sid", stub, _noop_on_event, _noop_on_fatal)
            await session.close()
            assert stub.closed is True
            # 2回目は何も起きない（例外を投げない）ことを確認する
            await session.close()

        asyncio.run(scenario())
