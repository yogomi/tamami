"""WebRTCセッション（1接続分）の音声受信とストリーミング認識器への接続.

クライアントのofferを受けてRTCPeerConnectionを確立し、受信したOpus音声を
デコード・16kHz / monoへリサンプリング・float32正規化してStreamingRecognizerへ
feedする。認識器が生成するAsrEventはon_eventへ、過負荷・内部エラーなどの致命的
状態はon_fatalへ通知する（実際の配信・切断処理はsrc/server/app.py側が担う）。
"""

import asyncio
import logging
from typing import Awaitable, Callable, Optional

import numpy as np
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av.audio.resampler import AudioResampler

from src.speech.streaming import AsrEvent, RecognizerOverloadedError, StreamingRecognizer

logger = logging.getLogger(__name__)

# ASRに渡す音声形式（PROTOCOL.md「上り: 音声」を参照）
TARGET_SAMPLE_RATE = 16000

# close()がdispatch_taskの自然終了（キュー残余の送信完了）を待つ最大秒数。
# WebSocket送信がハングした場合に無期限ブロックしないための保険。
DISPATCH_DRAIN_TIMEOUT_SEC = 5.0

# 認識結果イベントの通知コールバック
AsrEventCallback = Callable[[AsrEvent], Awaitable[None]]
# 致命的エラーの通知コールバック（引数はエラーコード, メッセージ。PROTOCOL.md参照）
FatalCallback = Callable[[str, str], Awaitable[None]]


class StreamingSession:
    """1つのWebSocket接続に紐づくWebRTCセッション.

    Args:
        session_id: セッション識別子（ログ用）。
        recognizer: 音声をfeedするストリーミング認識器。
        on_event: 認識結果イベントの通知先コールバック。
        on_fatal: 致命的エラーの通知先コールバック。
            引数は(エラーコード, メッセージ)。

    Attributes:
        received_seconds: 音声タイムライン上の受信済み秒数
            （最初に受信したサンプルを0とする）。
    """

    def __init__(
        self,
        session_id: str,
        recognizer: StreamingRecognizer,
        on_event: AsrEventCallback,
        on_fatal: FatalCallback,
    ) -> None:
        """セッションを初期化する.

        Args:
            session_id: セッション識別子。
            recognizer: 音声をfeedするストリーミング認識器。
            on_event: 認識結果イベントの通知先コールバック。
            on_fatal: 致命的エラーの通知先コールバック。
        """
        self._session_id = session_id
        self._recognizer = recognizer
        self._on_event = on_event
        self._on_fatal = on_fatal
        self._pc: Optional[RTCPeerConnection] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._samples_received = 0
        self._closed = False
        # 認識結果の配送はwebrtc_offer前・音声未到着でも始めてよいため、
        # 生成直後から起動しておく。
        self._dispatch_task: Optional[asyncio.Task] = asyncio.ensure_future(self._dispatch_events())

    @property
    def received_seconds(self) -> float:
        """音声タイムライン上の受信済み秒数を返す.

        Returns:
            受信済みサンプル数を16kHz換算した秒数。
        """
        return self._samples_received / TARGET_SAMPLE_RATE

    async def handle_offer(self, sdp: str) -> str:
        """クライアントのofferを処理し、answerのSDPを返す.

        RTCPeerConnectionを作成して音声トラックの受信を開始する。
        aiortcはsetLocalDescription内でICE候補の収集完了を待つため、
        返されるSDPは候補を含む（non-trickle）。

        Args:
            sdp: クライアントから受信したofferのSDP。

        Returns:
            ICE候補を含むanswerのSDP。

        Raises:
            ValueError: SDPの解釈に失敗した場合（aiortc由来）。
        """
        # ICEサーバーは設定しない（LAN内のホスト候補のみで接続する）。
        # aiortcのデフォルトはGoogleのSTUNサーバーで、aioiceがそのDNS解決を
        # タイムアウトなしのexecutorジョブとして実行するため、DNSが応答しない
        # 環境ではスレッドが残り続けプロセス終了時のjoinが固まる。また収集完了を
        # setLocalDescription内で待つため、answer生成の遅延要因にもなる。
        # NAT越えが必要になったら、STUN / TURNはIPアドレス指定で設定すること。
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self._pc = pc

        @pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if track.kind != "audio":
                logger.warning("[%s] ignoring non-audio track: %s", self._session_id, track.kind)
                return
            logger.info("[%s] audio track received", self._session_id)
            self._consumer_task = asyncio.ensure_future(self._consume(track))

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return pc.localDescription.sdp

    async def _consume(self, track: MediaStreamTrack) -> None:
        """音声トラックを消費し、リサンプリング・正規化して認識器へfeedする.

        Args:
            track: 受信した音声トラック。

        副作用:
            受信のたびにreceived_secondsを更新し、recognizer.feed()を呼ぶ。
        """
        resampler = AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)
        try:
            while True:
                try:
                    frame = await track.recv()
                except MediaStreamError:
                    logger.info("[%s] audio track ended", self._session_id)
                    break
                for resampled in resampler.resample(frame):
                    pcm_s16 = resampled.to_ndarray().reshape(-1)
                    self._samples_received += len(pcm_s16)
                    # int16 -> float32 [-1.0, 1.0] への正規化（変換点はここ1箇所）
                    pcm = pcm_s16.astype(np.float32) / 32768.0
                    self._recognizer.feed(pcm)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[%s] audio consumer failed", self._session_id)

    async def _dispatch_events(self) -> None:
        """recognizer.events()を消費し、on_event/on_fatalへ振り分ける.

        RecognizerOverloadedErrorはon_fatal("overloaded", ...)へ、
        それ以外の例外はon_fatal("internal_error", ...)へ写像する。
        """
        try:
            async for event in self._recognizer.events():
                await self._on_event(event)
        except asyncio.CancelledError:
            raise
        except RecognizerOverloadedError as e:
            logger.warning("[%s] recognizer overloaded: %s", self._session_id, e)
            await self._on_fatal("overloaded", str(e))
        except Exception as e:
            logger.exception("[%s] recognizer failed", self._session_id)
            await self._on_fatal("internal_error", str(e))

    async def flush(self) -> None:
        """処理中の音声をフラッシュし、確定結果を出し切るまで待つ.

        session_end受信時に呼ぶ（PROTOCOL.md「残りの確定結果をすべて送ってから
        閉じる」）。recognizer.flush()のドレイン契約にそのまま委譲する。
        """
        await self._recognizer.flush()

    async def close(self) -> None:
        """受信タスク・認識器・イベント配送タスク・RTCPeerConnectionを停止する.

        PROTOCOL.mdの「session_end時、残りの確定結果をすべて送ってから閉じる」
        を満たすため、_dispatch_taskは即座にcancelしない。手順は次のとおり:

        1. _consumer_taskをcancelし、これ以上音声がfeedされないようにする
        2. recognizer.close()をawaitする。これによりワーカースレッドが停止し、
           終端マーカー（またはエラーマーカー）がevents()側のasyncio.Queueの
           末尾に積まれる。この時点で、それ以前に生成された確定イベントも
           すべて同じキューに（順序を保って）積まれていることが保証される
        3. _dispatch_taskは、2で積まれたイベントをキューから取り出しon_event
           （WebSocket送信）で送り切ってから自然終了するのを待つ。cancelは
           しない（cancelすると送信途中の確定イベントが失われ得るため）

        ただし3でWebSocket送信自体がハングした場合に無期限ブロックしないよう、
        自然終了はDISPATCH_DRAIN_TIMEOUT_SEC秒だけ待ち、超過時のみcancelに
        フォールバックする。

        複数回呼んでも安全（冪等）。
        """
        if self._closed:
            return
        self._closed = True

        if self._consumer_task is not None:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None

        await self._recognizer.close()

        if self._dispatch_task is not None:
            try:
                await asyncio.wait_for(self._dispatch_task, timeout=DISPATCH_DRAIN_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                logger.warning(
                    "[%s] dispatch task did not drain within %.1fs, cancelling",
                    self._session_id,
                    DISPATCH_DRAIN_TIMEOUT_SEC,
                )
                self._dispatch_task.cancel()
                try:
                    await self._dispatch_task
                except asyncio.CancelledError:
                    pass
            self._dispatch_task = None

        if self._pc is not None:
            await self._pc.close()
            self._pc = None
