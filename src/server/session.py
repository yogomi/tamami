"""WebRTC セッション（1 接続分）の音声受信と状態管理.

クライアントの offer を受けて RTCPeerConnection を確立し、受信した Opus 音声を
デコード・16kHz / mono へリサンプリングして音声タイムラインを集計する。

現段階は疎通確認用のエコー実装であり、一定間隔ごとに受信秒数と音声レベルを
レポートとして通知する。ASR 接続時にこのレポート部分を認識結果に置き換える。
"""

import asyncio
import logging
import math
from typing import Awaitable, Callable, Optional

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av.audio.resampler import AudioResampler

logger = logging.getLogger(__name__)

# ASR に渡す音声形式（PROTOCOL.md「上り: 音声」を参照）
TARGET_SAMPLE_RATE = 16000

# エコーレポートの集計間隔（受信音声の秒数ベース）
REPORT_INTERVAL_SEC = 0.5

# レポート内容: 受信済み秒数と音声レベル
EchoReport = dict[str, float]

# レポート通知コールバック（is_final, ts_audio_end, level_db を受け取る）
ReportCallback = Callable[[bool, float, float], Awaitable[None]]


class StreamingSession:
    """1 つの WebSocket 接続に紐づく WebRTC セッション.

    Args:
        session_id: セッション識別子（ログ用）。
        on_report: 集計レポートの通知先コールバック。
            引数は (is_final, ts_audio_end, level_db)。

    Attributes:
        received_seconds: 音声タイムライン上の受信済み秒数
            （最初に受信したサンプルを 0 とする）。
    """

    def __init__(self, session_id: str, on_report: ReportCallback) -> None:
        """セッションを初期化する.

        Args:
            session_id: セッション識別子。
            on_report: 集計レポートの通知先コールバック。
        """
        self._session_id = session_id
        self._on_report = on_report
        self._pc: Optional[RTCPeerConnection] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._samples_received = 0
        self._window_sum_squares = 0.0
        self._window_samples = 0

    @property
    def received_seconds(self) -> float:
        """音声タイムライン上の受信済み秒数を返す.

        Returns:
            受信済みサンプル数を 16kHz 換算した秒数。
        """
        return self._samples_received / TARGET_SAMPLE_RATE

    async def handle_offer(self, sdp: str) -> str:
        """クライアントの offer を処理し、answer の SDP を返す.

        RTCPeerConnection を作成して音声トラックの受信を開始する。
        aiortc は setLocalDescription 内で ICE 候補の収集完了を待つため、
        返される SDP は候補を含む（non-trickle）。

        Args:
            sdp: クライアントから受信した offer の SDP。

        Returns:
            ICE 候補を含む answer の SDP。

        Raises:
            ValueError: SDP の解釈に失敗した場合（aiortc 由来）。
        """
        pc = RTCPeerConnection()
        self._pc = pc

        @pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if track.kind != "audio":
                logger.warning(
                    "[%s] ignoring non-audio track: %s", self._session_id, track.kind
                )
                return
            logger.info("[%s] audio track received", self._session_id)
            self._consumer_task = asyncio.ensure_future(self._consume(track))

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return pc.localDescription.sdp

    async def _consume(self, track: MediaStreamTrack) -> None:
        """音声トラックを消費し、リサンプリングして集計する.

        Args:
            track: 受信した音声トラック。

        副作用:
            REPORT_INTERVAL_SEC 分の音声を受信するごとに on_report を呼ぶ。
        """
        resampler = AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)
        last_report_samples = 0
        report_interval_samples = int(REPORT_INTERVAL_SEC * TARGET_SAMPLE_RATE)
        try:
            while True:
                try:
                    frame = await track.recv()
                except MediaStreamError:
                    logger.info("[%s] audio track ended", self._session_id)
                    break
                for resampled in resampler.resample(frame):
                    pcm = resampled.to_ndarray().reshape(-1).astype(np.float64)
                    self._samples_received += len(pcm)
                    self._window_sum_squares += float(np.sum((pcm / 32768.0) ** 2))
                    self._window_samples += len(pcm)
                if (
                    self._samples_received - last_report_samples
                    >= report_interval_samples
                ):
                    last_report_samples = self._samples_received
                    await self._on_report(
                        False, self.received_seconds, self._window_level_db()
                    )
                    self._window_sum_squares = 0.0
                    self._window_samples = 0
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[%s] audio consumer failed", self._session_id)

    def _window_level_db(self) -> float:
        """現在の集計ウィンドウの音声レベルを dBFS で返す.

        Returns:
            RMS レベル（dBFS）。無音・未受信時は -120.0。
        """
        if self._window_samples == 0:
            return -120.0
        rms = math.sqrt(self._window_sum_squares / self._window_samples)
        if rms <= 0.0:
            return -120.0
        return 20.0 * math.log10(rms)

    async def flush(self) -> None:
        """処理中の音声をフラッシュし、確定レポートを通知する.

        session_end 受信時・切断時に呼ぶ。エコー実装では受信合計の
        確定レポート（is_final: True）を 1 回送る。
        """
        if self._samples_received > 0:
            await self._on_report(True, self.received_seconds, self._window_level_db())

    async def close(self) -> None:
        """受信タスクと RTCPeerConnection を停止する."""
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None
        if self._pc is not None:
            await self._pc.close()
            self._pc = None
