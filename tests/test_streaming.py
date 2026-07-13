"""speech.streaming / speech.fake モジュールのテスト."""

import asyncio

import numpy as np
import pytest

from src.speech.fake import FakeStreamingRecognizer
from src.speech.streaming import (
    SAMPLE_RATE,
    AsrEvent,
    EndpointDecision,
    RecognizerOverloadedError,
    SilenceEndpointer,
    StreamingRecognizer,
)


def _tone(seconds: float, amplitude: float = 0.5) -> np.ndarray:
    """テスト用の正弦波PCM（発話の代わり）を生成する.

    Args:
        seconds: 長さ（秒）。
        amplitude: 振幅（0.0〜1.0）。

    Returns:
        float32のPCM。
    """
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    """テスト用の無音PCM（ゼロ埋め）を生成する.

    Args:
        seconds: 長さ（秒）。

    Returns:
        float32のPCM（すべて0）。
    """
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


class TestSilenceEndpointer:
    """SilenceEndpointer単体のテスト."""

    def test_speech_while_loud(self):
        endpointer = SilenceEndpointer(silence_threshold_db=-40.0, min_silence_sec=0.2)
        assert endpointer.process(_tone(0.1)) == EndpointDecision.SPEECH

    def test_silence_before_speech(self):
        endpointer = SilenceEndpointer(silence_threshold_db=-40.0, min_silence_sec=0.2)
        assert endpointer.process(_silence(0.1)) == EndpointDecision.SILENCE

    def test_segment_end_after_min_silence(self):
        endpointer = SilenceEndpointer(silence_threshold_db=-40.0, min_silence_sec=0.2)
        assert endpointer.process(_tone(0.1)) == EndpointDecision.SPEECH
        # min_silence_sec未満の無音はまだ終端としない
        assert endpointer.process(_silence(0.1)) == EndpointDecision.SILENCE
        # 累計でmin_silence_secに達すると終端
        assert endpointer.process(_silence(0.1)) == EndpointDecision.SEGMENT_END

    def test_silence_after_segment_end_is_silence_not_end(self):
        endpointer = SilenceEndpointer(silence_threshold_db=-40.0, min_silence_sec=0.1)
        assert endpointer.process(_tone(0.1)) == EndpointDecision.SPEECH
        assert endpointer.process(_silence(0.1)) == EndpointDecision.SEGMENT_END
        # 発話中でなくなった後の無音は、SEGMENT_ENDを繰り返さずSILENCE
        assert endpointer.process(_silence(0.1)) == EndpointDecision.SILENCE

    def test_force_segment_end(self):
        endpointer = SilenceEndpointer()
        assert endpointer.force_segment_end() is False
        endpointer.process(_tone(0.1))
        assert endpointer.force_segment_end() is True
        assert endpointer.force_segment_end() is False


class TestFakeStreamingRecognizerSemantics:
    """FakeStreamingRecognizerの発話→暫定→無音→確定のセマンティクステスト."""

    def test_full_lifecycle(self):
        async def scenario():
            recognizer = FakeStreamingRecognizer(min_silence_sec=0.5)
            it = recognizer.events()
            try:
                # 発話開始: 暫定イベント(is_final=False)がsegment_id=0で届く
                recognizer.feed(_tone(0.5))
                ev1 = await asyncio.wait_for(it.__anext__(), timeout=2.0)
                assert ev1.is_final is False
                assert ev1.segment_id == 0
                assert ev1.lang == "ja"
                assert "0.5" in ev1.text

                # 発話継続: 全文が置き換わり、秒数が伸びる
                recognizer.feed(_tone(0.5))
                ev2 = await asyncio.wait_for(it.__anext__(), timeout=2.0)
                assert ev2.is_final is False
                assert ev2.segment_id == 0
                assert "1.0" in ev2.text

                # 無音継続でセグメント終端 -> is_final=True
                recognizer.feed(_silence(0.5))
                ev3 = await asyncio.wait_for(it.__anext__(), timeout=2.0)
                assert ev3.is_final is True
                assert ev3.segment_id == 0

                # 次の発話でsegment_idが増える
                recognizer.feed(_tone(0.5))
                ev4 = await asyncio.wait_for(it.__anext__(), timeout=2.0)
                assert ev4.is_final is False
                assert ev4.segment_id == 1
            finally:
                await recognizer.close()

        asyncio.run(scenario())

    def test_text_is_deterministic(self):
        """同じ音声を2回処理すれば同じテキストになる（乱数不使用）."""

        async def scenario():
            texts = []
            for _ in range(2):
                recognizer = FakeStreamingRecognizer(min_silence_sec=0.5)
                it = recognizer.events()
                recognizer.feed(_tone(0.5, amplitude=0.5))
                ev = await asyncio.wait_for(it.__anext__(), timeout=2.0)
                texts.append(ev.text)
                await recognizer.close()
            assert texts[0] == texts[1]

        asyncio.run(scenario())


class TestFlushDrainContract:
    """flush()のドレイン契約のテスト."""

    def test_flush_emits_final_before_returning(self):
        async def scenario():
            recognizer = FakeStreamingRecognizer(min_silence_sec=0.5)
            recognizer.feed(_tone(1.0))
            await recognizer.flush()

            it = recognizer.events()
            collected: list[AsrEvent] = []
            for _ in range(5):
                ev = await asyncio.wait_for(it.__anext__(), timeout=1.0)
                collected.append(ev)
                if ev.is_final:
                    break

            assert collected, "flush後に少なくとも1件のイベントが届くはず"
            assert collected[-1].is_final is True
            await recognizer.close()

        asyncio.run(scenario())


class _FailingOnFinalizeRecognizer(StreamingRecognizer):
    """_finalize()が必ず例外を投げるテスト用サブクラス."""

    def _process(self, pcm: np.ndarray) -> list[AsrEvent]:
        return []

    def _finalize(self) -> list[AsrEvent]:
        raise RuntimeError("finalize boom")


class _FailingOnProcessRecognizer(StreamingRecognizer):
    """_process()が必ず例外を投げるテスト用サブクラス."""

    def _process(self, pcm: np.ndarray) -> list[AsrEvent]:
        raise RuntimeError("process boom")

    def _finalize(self) -> list[AsrEvent]:
        return []


class TestFlushDoesNotHang:
    """flush()がワーカー死亡時にハングしないことのレグレッションテスト.

    コードレビュー指摘: 以前はワーカーが例外死した場合・flush待ち中にcloseされた
    場合にflush()側のasyncio.Eventが誰にもsetされず永久にハングしていた。
    """

    def test_flush_does_not_hang_when_finalize_raises(self):
        async def scenario():
            recognizer = _FailingOnFinalizeRecognizer()
            recognizer.feed(np.zeros(10, dtype=np.float32))

            # _finalize()が例外を投げても、flush()自体はハングせず返るはず
            await asyncio.wait_for(recognizer.flush(), timeout=2.0)

            it = recognizer.events()
            with pytest.raises(RuntimeError, match="finalize boom"):
                await asyncio.wait_for(it.__anext__(), timeout=2.0)

            await recognizer.close()

        asyncio.run(scenario())

    def test_flush_after_worker_already_dead_does_not_hang(self):
        async def scenario():
            recognizer = _FailingOnProcessRecognizer()
            it = recognizer.events()
            recognizer.feed(np.zeros(10, dtype=np.float32))

            # ワーカーは_process()の例外で死ぬ。events()側から再送出される
            with pytest.raises(RuntimeError, match="process boom"):
                await asyncio.wait_for(it.__anext__(), timeout=2.0)

            # ワーカー死亡後にflush()を呼んでも、待たされずに即returnするはず
            await asyncio.wait_for(recognizer.flush(), timeout=2.0)
            await recognizer.close()

        asyncio.run(scenario())

    def test_flush_and_close_concurrently_do_not_hang(self):
        async def scenario():
            recognizer = FakeStreamingRecognizer(min_silence_sec=0.5)
            recognizer.feed(_tone(0.1))

            # flush()とclose()を同時に走らせ、どちらの順序でも双方が返ることを
            # 確認する（flush待ち中にワーカーが停止するケースの網羅）
            flush_task = asyncio.create_task(recognizer.flush())
            close_task = asyncio.create_task(recognizer.close())
            await asyncio.wait_for(asyncio.gather(flush_task, close_task), timeout=3.0)

        asyncio.run(scenario())


class TestCloseSemantics:
    """close()の終端・冪等性・feed禁止のテスト."""

    def test_close_terminates_events_iterator(self):
        async def scenario():
            recognizer = FakeStreamingRecognizer()
            recognizer.feed(_tone(0.1))
            await recognizer.close()
            events = [ev async for ev in recognizer.events()]
            # closeにより即座に終端するため、この時点でイテレーションは空か
            # ごく少数で完了する（例外を投げずに完了することが重要）
            assert isinstance(events, list)

        asyncio.run(scenario())

    def test_close_is_idempotent(self):
        async def scenario():
            recognizer = FakeStreamingRecognizer()
            await recognizer.close()
            await recognizer.close()

        asyncio.run(scenario())

    def test_feed_after_close_raises(self):
        async def scenario():
            recognizer = FakeStreamingRecognizer()
            await recognizer.close()
            with pytest.raises(RuntimeError):
                recognizer.feed(_tone(0.1))

        asyncio.run(scenario())


class TestBackpressure:
    """バックプレッシャ（RecognizerOverloadedError）のテスト."""

    def test_overloaded_when_input_queue_exceeds_limit(self):
        async def scenario():
            recognizer = FakeStreamingRecognizer(
                max_buffered_seconds=0.3,
                processing_delay=0.5,
            )
            it = recognizer.events()
            # ワーカーは最初のチャンクの処理でprocessing_delay分眠るため、
            # その間に大量の音声を積むとキュー滞留がmax_buffered_secondsを超える
            for _ in range(10):
                recognizer.feed(_tone(0.1))

            with pytest.raises(RecognizerOverloadedError):
                await asyncio.wait_for(it.__anext__(), timeout=3.0)

            await recognizer.close()

        asyncio.run(scenario())


class _RecordingRecognizer(StreamingRecognizer):
    """チャンク再バッファ機構を検証するための記録用サブクラス."""

    def __init__(self, chunk_samples: int) -> None:
        super().__init__(chunk_samples=chunk_samples)
        self.received_lengths: list[int] = []

    def _process(self, pcm: np.ndarray) -> list[AsrEvent]:
        self.received_lengths.append(len(pcm))
        return []

    def _finalize(self) -> list[AsrEvent]:
        return []


class TestChunkRebuffering:
    """基底クラスのチャンク再バッファ機構のテスト."""

    def test_feed_is_resplit_into_fixed_chunks(self):
        async def scenario():
            recognizer = _RecordingRecognizer(chunk_samples=200)
            recognizer.feed(np.zeros(100, dtype=np.float32))
            recognizer.feed(np.zeros(250, dtype=np.float32))
            recognizer.feed(np.zeros(50, dtype=np.float32))
            await recognizer.flush()
            # 100 + 250 + 50 = 400 samples はぴったり200*2個に分割される
            assert recognizer.received_lengths == [200, 200]
            await recognizer.close()

        asyncio.run(scenario())

    def test_leftover_is_processed_before_finalize(self):
        async def scenario():
            recognizer = _RecordingRecognizer(chunk_samples=200)
            recognizer.feed(np.zeros(100, dtype=np.float32))
            await recognizer.flush()
            # 端数(100 < chunk_samples)はflush時に_processへ渡される
            assert recognizer.received_lengths == [100]
            await recognizer.close()

        asyncio.run(scenario())
