"""ストリーミングASR認識器の抽象基底クラスと共通部品.

サブクラス（FakeStreamingRecognizer・NemotronStreamingRecognizer）は推論フック
（_process / _finalize）だけを実装すればよく、ワーカースレッドの起動・入力の
バッファリング・非同期イベント配送・バックプレッシャ検出は本モジュールの
StreamingRecognizerが提供する。

StreamingRecognizerのインスタンスは、実行中のasyncioイベントループの中で
生成すること（events()の配送にloop.call_soon_threadsafeを使うため、
生成時点でasyncio.get_running_loop()が成立している必要がある）。
"""

import asyncio
import logging
import math
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ASRに渡す音声のサンプルレート（PROTOCOL.md「上り: 音声」を参照）。
# AsrEventのts_audio_*・max_buffered_secondsの換算に用いる。
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AsrEvent:
    """認識結果1件分のイベント.

    Attributes:
        segment_id: セッション内で単調増加するセグメント番号（認識器が採番する）。
        text: セグメントの認識テキスト全文（差分ではなく、常に全文で置き換える）。
        lang: 判定された言語コード（例: "uk" / "ja"）。
        is_final: セグメントの認識が確定したかどうか。
        ts_audio_start: 音声タイムライン上のセグメント開始秒
            （feedされたサンプル数からSAMPLE_RATE換算で算出）。
        ts_audio_end: 音声タイムライン上のセグメント終了秒（同上）。
    """

    segment_id: int
    text: str
    lang: str
    is_final: bool
    ts_audio_start: float
    ts_audio_end: float


class RecognizerOverloadedError(Exception):
    """入力キューが上限を超え、推論がリアルタイムに追いつかない状態を表す.

    cache-awareストリーミング認識は音声を間引いて追いつく設計にできないため、
    一度この状態になったセッションには回復の道がない。呼び出し側はセッションを
    エラー終了させる想定（PROTOCOL.mdのerror: overloaded）。
    """


class EndpointDecision(Enum):
    """SilenceEndpointer.process()が返す発話区間の判定."""

    SPEECH = "speech"
    """発話中（無音閾値を超える音声が続いている）。"""

    SILENCE = "silence"
    """無音継続中（まだ終端未確定、または発話開始前）。"""

    SEGMENT_END = "segment_end"
    """発話区間の終端を検出した（このチャンクでセグメントを確定してよい）。"""


class SilenceEndpointer:
    """無音ベースのセグメント終端（endpoint）判定器.

    PCMチャンクを順に受け取り、RMSレベル（dBFS）が閾値以下の無音状態が
    min_silence_sec続いたら発話区間の終端とみなす。Nemotronの実発話終端検出
    （EOU）やSilero VADなど、より精度の高い判定器へ差し替え可能なよう、
    StreamingRecognizerとは独立した部品として設計する
    （STREAMING_PLAN.mdの通り、この判定がE2E遅延の支配項になり得るため）。

    Args:
        silence_threshold_db: 無音とみなすRMSレベルの閾値（dBFS）。
        min_silence_sec: この秒数だけ無音が継続したら終端とみなす。

    Attributes:
        なし（すべて非公開状態として保持する）。
    """

    def __init__(
        self,
        silence_threshold_db: float = -40.0,
        min_silence_sec: float = 0.5,
    ) -> None:
        """判定器を初期化する.

        Args:
            silence_threshold_db: 無音とみなすRMSレベルの閾値（dBFS）。
            min_silence_sec: 終端とみなすまでに必要な無音継続秒数。
        """
        self._silence_threshold_db = silence_threshold_db
        self._min_silence_sec = min_silence_sec
        self._speaking = False
        self._silence_sec = 0.0

    def process(self, pcm: np.ndarray) -> EndpointDecision:
        """1チャンク分のPCMを判定する.

        Args:
            pcm: float32・[-1.0, 1.0]のPCMチャンク（任意長）。

        Returns:
            発話中はSPEECH、無音継続中（終端未確定）はSILENCE、
            発話区間の終端を検出した場合はSEGMENT_END。
        """
        level_db = rms_dbfs(pcm)
        chunk_sec = len(pcm) / SAMPLE_RATE
        if level_db > self._silence_threshold_db:
            self._speaking = True
            self._silence_sec = 0.0
            return EndpointDecision.SPEECH

        if not self._speaking:
            return EndpointDecision.SILENCE

        self._silence_sec += chunk_sec
        if self._silence_sec >= self._min_silence_sec:
            self._speaking = False
            self._silence_sec = 0.0
            return EndpointDecision.SEGMENT_END
        return EndpointDecision.SILENCE

    def force_segment_end(self) -> bool:
        """発話中であれば強制的に終端させる（flush/close時に使用）.

        音声の後続が来ないままセッションが終わる場合、無音の継続を待たずに
        セグメントを閉じるためにサブクラスの_finalize()から呼ぶ想定。

        Returns:
            発話中だった場合True（呼び出し側は終端イベントを生成してよい）。
            発話中でなければ何もせずFalseを返す。
        """
        if self._speaking:
            self._speaking = False
            self._silence_sec = 0.0
            return True
        return False


def rms_dbfs(pcm: np.ndarray) -> float:
    """PCMチャンクのRMSレベルをdBFSで返す.

    Args:
        pcm: float32・[-1.0, 1.0]のPCMチャンク。

    Returns:
        RMSレベル（dBFS）。空配列・無音時は-120.0。
    """
    if len(pcm) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))))
    if rms <= 0.0:
        return -120.0
    return 20.0 * math.log10(rms)


class _FlushRequest:
    """flush()の内部マーカー. ワーカーが処理完了後にdoneイベントを立てる."""

    def __init__(self, done: asyncio.Event) -> None:
        """マーカーを初期化する.

        Args:
            done: ワーカーが処理を終えたことを通知するasyncio.Event。
        """
        self.done = done


class _WorkerFinished:
    """ワーカースレッドが正常終了したことを示す内部マーカー."""


class _WorkerFailed:
    """ワーカースレッドで例外が発生したことを示す内部マーカー."""

    def __init__(self, exc: BaseException) -> None:
        """マーカーを初期化する.

        Args:
            exc: ワーカースレッドで発生した例外（events()側で再送出する）。
        """
        self.exc = exc


_InputItem = Union[np.ndarray, _FlushRequest]
_EventQueueItem = Union[AsrEvent, _WorkerFinished, _WorkerFailed]


class StreamingRecognizer:
    """ストリーミング音声認識器の抽象基底クラス.

    サブクラスは推論フック（_process / _finalize）だけを実装すればよい。
    本クラスがワーカースレッドの起動・入力バッファリング・チャンク再分割・
    非同期イベント配送・バックプレッシャ検出を提供する。

    Args:
        max_buffered_seconds: 入力キューに滞留してよい未処理音声の秒数上限。
            これを超えると過負荷状態とみなし、以後events()が
            RecognizerOverloadedErrorを送出する（cache-awareストリーミングは
            音声を間引いて追いつく設計にできないため、回復の道はない）。
        chunk_samples: 指定した場合、feed()された任意長のPCMをこのサンプル数の
            固定長チャンクに切り直してから_processに渡す（端数は次チャンクへ
            持ち越し、flush/close時の_finalize前に残余を処理する）。
            Noneの場合はfeed()された単位のまま_processに渡す。

    Attributes:
        なし（すべて非公開状態として保持する）。

    サブクラスが実装するフック（ワーカースレッド上で呼ばれる同期メソッド）:
        _process(pcm: np.ndarray) -> list[AsrEvent]: 1チャンクを処理する。
        _finalize() -> list[AsrEvent]: flush/close時、進行中セグメントを確定する。
    """

    def __init__(
        self,
        max_buffered_seconds: float = 10.0,
        chunk_samples: Optional[int] = None,
    ) -> None:
        """認識器を初期化し、ワーカースレッドを起動する.

        呼び出し時点で実行中のasyncioイベントループが必要（events()の配送に
        loop.call_soon_threadsafeを使うため）。

        Args:
            max_buffered_seconds: バックプレッシャ検出の閾値（秒）。
            chunk_samples: チャンク再バッファのサイズ（サンプル数）。Noneなら無効。

        Raises:
            RuntimeError: 実行中のイベントループがない場合
                （asyncio.get_running_loopに由来）。
        """
        self._max_buffered_samples = int(max_buffered_seconds * SAMPLE_RATE)
        self._chunk_samples = chunk_samples
        self._carry: Optional[np.ndarray] = None  # チャンク再バッファの端数

        self._input_lock = threading.Lock()
        self._input_cond = threading.Condition(self._input_lock)
        self._input_items: "deque[_InputItem]" = deque()
        self._buffered_samples = 0
        self._stop_requested = False
        # ワーカースレッドが終了した（正常終了・例外死のいずれか）ことを示す。
        # _input_cond配下で管理し、flush()が「append + notify」と同一クリティカル
        # セクションでこのフラグを確認できるようにする（flush()の永久ハング防止）。
        self._worker_stopped = False

        self._overloaded = False
        self._closed = False
        self._close_lock = threading.Lock()

        self._loop = asyncio.get_running_loop()
        self._event_queue: "asyncio.Queue[_EventQueueItem]" = asyncio.Queue()

        self._worker = threading.Thread(
            target=self._run_worker, name="asr-recognizer-worker", daemon=True
        )
        self._worker.start()

    # --- サブクラスが実装するフック ---------------------------------------

    def _process(self, pcm: np.ndarray) -> list[AsrEvent]:
        """1チャンク分のPCMを処理する（ワーカースレッド上で呼ばれる）.

        Args:
            pcm: chunk_samples長（または未指定時はfeedされた単位）のPCM。

        Returns:
            このチャンクの処理で生成されたAsrEventのリスト。
        """
        raise NotImplementedError

    def _finalize(self) -> list[AsrEvent]:
        """進行中セグメントを確定する（ワーカースレッド上で呼ばれる）.

        flush()・close()時に、チャンク再バッファの残余処理の後で呼ばれる。

        Returns:
            確定処理で生成されたAsrEventのリスト（進行中セグメントがなければ空）。
        """
        raise NotImplementedError

    # --- 公開API -----------------------------------------------------------

    def feed(self, pcm: np.ndarray) -> None:
        """PCMを非ブロッキングで入力キューに積む.

        Args:
            pcm: 16kHz / mono / float32 / [-1.0, 1.0] の任意長PCM。

        Raises:
            RuntimeError: close()後に呼ばれた場合。

        副作用:
            キュー内の未処理音声がmax_buffered_secondsを超えたら過負荷状態と
            なり、以後の呼び出しは無視される（events()がRecognizerOverloadedError
            を送出する）。
        """
        if self._closed:
            raise RuntimeError("feed() called after close()")
        if self._overloaded:
            # 既に過負荷状態。回復の道がないため、これ以上取り込まない。
            return
        with self._input_cond:
            self._input_items.append(pcm)
            self._buffered_samples += len(pcm)
            exceeded = self._buffered_samples > self._max_buffered_samples
            self._input_cond.notify()
        if exceeded:
            self._overloaded = True
            self._event_queue.put_nowait(
                _WorkerFailed(
                    RecognizerOverloadedError(
                        f"input queue exceeded {self._max_buffered_samples} "
                        "buffered samples (max_buffered_seconds)"
                    )
                )
            )

    async def events(self) -> AsyncIterator[AsrEvent]:
        """認識イベントを順に取り出す非同期イテレータ（pull型）.

        ワーカースレッドが生成したAsrEventをasyncio.Queue経由で順序保証つきに
        配送する。ワーカースレッドで発生した例外（RecognizerOverloadedErrorを
        含む）はこのイテレータから再送出する。close()完了で正常終端する。

        Yields:
            認識結果イベント。

        Raises:
            RecognizerOverloadedError: 入力キューが過負荷状態になった場合。
            Exception: _process/_finalizeで発生した例外。
        """
        while True:
            item = await self._event_queue.get()
            if isinstance(item, _WorkerFinished):
                return
            if isinstance(item, _WorkerFailed):
                raise item.exc
            yield item

    async def flush(self) -> None:
        """入力キュー残余をすべて処理し、進行中セグメントを確定させるまで待つ.

        session_end時に呼ぶ（PROTOCOL.mdの「残りの確定結果をすべて送ってから
        閉じる」を満たすためのドレイン契約）。ワーカーが正常に稼働している場合、
        この呼び出しが完了した時点で、進行中セグメントのis_finalイベントは
        events()側のキューに積まれている。

        この呼び出しは**ワーカーの生死によらず必ず返る**（ハングしないことを
        保証する）。ワーカーが既に停止している（例外死・close()済みのいずれか）
        場合は待たずに直ちに返り、ドレイン処理は行われない。ワーカーが処理中に
        例外を投げて死んだ場合も、そのFlushRequestのdoneは_run_workerのfinally
        で必ずsetされるため待ちきりになる。いずれの異常系でも、失敗自体は
        events()側の_WorkerFailed経由で別途伝わる（flush()自体は失敗を通知しない）。

        Raises:
            RuntimeError: close()後に呼ばれた場合。
        """
        if self._closed:
            raise RuntimeError("flush() called after close()")
        done = asyncio.Event()
        with self._input_cond:
            if self._worker_stopped:
                # ワーカーは既に停止済み。誰もdoneをsetしないため待たずに返る。
                return
            self._input_items.append(_FlushRequest(done))
            self._input_cond.notify()
        await done.wait()

    async def close(self) -> None:
        """ワーカースレッドを停止し、資源を解放する.

        events()イテレータを終端させる。複数回呼んでも安全（冪等）。
        flush()と異なり、入力キューに残った未処理音声はドレインせず破棄する
        （資源解放のためのクローズであり、確定結果の取りこぼしはflush()を
        別途呼んで防ぐ想定）。
        """
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        with self._input_cond:
            self._stop_requested = True
            self._input_cond.notify_all()
        await asyncio.to_thread(self._worker.join)

    # --- ワーカースレッド内部 ------------------------------------------------

    def _worker_get_next(self) -> Optional[_InputItem]:
        """ワーカースレッド用: 次の入力アイテムを取得する.

        Returns:
            次のPCMまたはFlushRequest。停止要求時はNone
            （このとき未処理の入力・未処理のFlushRequestは_mark_worker_stopped
            側でまとめて回収・破棄される。FlushRequestのdoneは漏れなくsetされる）。
        """
        with self._input_cond:
            while not self._input_items and not self._stop_requested:
                self._input_cond.wait()
            if self._stop_requested:
                return None
            item = self._input_items.popleft()
            if isinstance(item, np.ndarray):
                self._buffered_samples -= len(item)
            return item

    def _dispatch(self, pcm: np.ndarray) -> list[AsrEvent]:
        """1件の入力PCMを、必要ならチャンク再バッファしてから_processへ渡す.

        Args:
            pcm: feed()された単位のPCM。

        Returns:
            _process呼び出し（0回以上）で生成されたAsrEventのリスト。
        """
        if self._chunk_samples is None:
            return self._process(pcm)

        self._carry = pcm if self._carry is None else np.concatenate([self._carry, pcm])
        events: list[AsrEvent] = []
        while len(self._carry) >= self._chunk_samples:
            chunk = self._carry[: self._chunk_samples]
            self._carry = self._carry[self._chunk_samples :]
            events.extend(self._process(chunk))
        return events

    def _drain_and_finalize(self) -> list[AsrEvent]:
        """flush時: チャンク再バッファの端数を処理してから_finalizeを呼ぶ.

        Returns:
            端数処理・確定処理で生成されたAsrEventのリスト。
        """
        events: list[AsrEvent] = []
        if self._chunk_samples is not None and self._carry is not None and len(self._carry) > 0:
            events.extend(self._process(self._carry))
            self._carry = None
        events.extend(self._finalize())
        return events

    def _emit_events(self, events: list[AsrEvent]) -> None:
        """生成したイベントをevents()側のasyncio.Queueへ橋渡しする.

        Args:
            events: このステップで生成されたAsrEventのリスト。
        """
        for event in events:
            self._loop.call_soon_threadsafe(self._event_queue.put_nowait, event)

    def _mark_worker_stopped(self) -> list[_FlushRequest]:
        """ワーカー停止をフラグに記録し、入力キューに残ったflush要求を回収する.

        _run_workerのfinallyから呼ぶ。停止後にflush()が待たずに即returnできる
        ようにフラグを立てるのと同一クリティカルセクションで、まだdoneが
        setされていないFlushRequest（正常処理される前に停止した分）を集める。
        取りこぼしを避けるため、入力キューはこの時点で空にする
        （停止後の残留PCMはどのみち処理されない）。

        Returns:
            まだdoneがsetされていないFlushRequestのリスト
            （doneのsetは呼び出し側の責務）。
        """
        with self._input_cond:
            self._worker_stopped = True
            pending_flushes = [
                item for item in self._input_items if isinstance(item, _FlushRequest)
            ]
            self._input_items.clear()
            self._buffered_samples = 0
        return pending_flushes

    def _run_worker(self) -> None:
        """ワーカースレッドのメインループ.

        入力キューからPCM・flush要求を順に取り出し、_process/_finalizeを呼ぶ。
        例外発生時・終了時は、それぞれ対応するマーカーをasyncio.Queueへ積んで
        events()を終端（または例外送出）させる。

        どのような終了経路（正常終了・例外死）でも、finallyでワーカー停止を
        記録し、入力キューに残っていたflush要求のdoneを漏れなくsetする
        （flush()が永久ハングしないことを保証するため。処理中の例外で
        _drain_and_finalize自体が失敗した場合のFlushRequestは、その場の
        try/finallyで個別にdoneをsetする）。
        """
        failure: Optional[BaseException] = None
        try:
            while True:
                item = self._worker_get_next()
                if item is None:
                    break
                if isinstance(item, _FlushRequest):
                    try:
                        events = self._drain_and_finalize()
                        self._emit_events(events)
                    finally:
                        self._loop.call_soon_threadsafe(item.done.set)
                    continue
                events = self._dispatch(item)
                self._emit_events(events)
        except Exception as exc:  # ワーカー例外はevents()側から再送出する
            logger.exception("recognizer worker failed")
            failure = exc
            self._loop.call_soon_threadsafe(self._event_queue.put_nowait, _WorkerFailed(exc))
        finally:
            pending_flushes = self._mark_worker_stopped()
            for pending in pending_flushes:
                self._loop.call_soon_threadsafe(pending.done.set)
            if failure is None:
                self._loop.call_soon_threadsafe(self._event_queue.put_nowait, _WorkerFinished())
