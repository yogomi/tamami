"""開発・テスト用の代役ストリーミング認識器.

本物のASR（NemotronStreamingRecognizer）はGPU必須でこのMacでは動かないため、
サーバー本体の疎通・テストにはFakeStreamingRecognizerを使う。ASRとしての精度は
持たず、SilenceEndpointerによる発話区間検出と、発話秒数・音声レベルから決定的に
生成したテキストで代替する（乱数は使わない）。
"""

import time

import numpy as np

from src.speech.streaming import (
    SAMPLE_RATE,
    AsrEvent,
    EndpointDecision,
    SilenceEndpointer,
    StreamingRecognizer,
    rms_dbfs,
)

# 発話中に暫定イベントをemitする間隔（秒）。chunk_samplesとして基底クラスへ渡す。
_INTERIM_INTERVAL_SEC = 0.5


class FakeStreamingRecognizer(StreamingRecognizer):
    """ASRの代役として動作する開発・テスト用のストリーミング認識器.

    SilenceEndpointerで発話区間を検出し、発話中は0.5秒ごとに暫定イベント
    （is_final: False、全文置き換え）を、終端検出時に確定イベント
    （is_final: True）をemitする。テキストは発話秒数・音声レベルから決定的に
    生成する（例: "(音声 3.2秒 / -25dB)"）。langは"ja"固定。

    Args:
        max_buffered_seconds: 基底クラス参照（バックプレッシャ検出の閾値）。
        processing_delay: _process呼び出しごとにtime.sleepする秒数。
            バックプレッシャ経路（RecognizerOverloadedError）のテスト用の
            遅延注入で、デフォルトは遅延なし。
        silence_threshold_db: SilenceEndpointerへ渡す無音判定閾値（dBFS）。
        min_silence_sec: SilenceEndpointerへ渡す終端判定に必要な無音継続秒数。
    """

    def __init__(
        self,
        max_buffered_seconds: float = 10.0,
        processing_delay: float = 0.0,
        silence_threshold_db: float = -40.0,
        min_silence_sec: float = 0.5,
    ) -> None:
        """認識器を初期化する.

        Args:
            max_buffered_seconds: バックプレッシャ検出の閾値（秒）。
            processing_delay: _process内で注入する遅延（秒）。
            silence_threshold_db: 無音判定のRMSレベル閾値（dBFS）。
            min_silence_sec: 終端とみなすまでに必要な無音継続秒数。
        """
        chunk_samples = int(_INTERIM_INTERVAL_SEC * SAMPLE_RATE)
        super().__init__(max_buffered_seconds=max_buffered_seconds, chunk_samples=chunk_samples)
        self._processing_delay = processing_delay
        self._endpointer = SilenceEndpointer(
            silence_threshold_db=silence_threshold_db, min_silence_sec=min_silence_sec
        )
        self._segment_id = 0
        self._segment_open = False
        self._segment_start_sample = 0
        self._samples_seen = 0
        self._last_level_db = -120.0

    def _make_text(self, level_db: float) -> str:
        """発話秒数・音声レベルから決定的にテキストを生成する.

        Args:
            level_db: 現在のセグメント末尾チャンクのRMSレベル（dBFS）。

        Returns:
            "(音声 {秒数}秒 / {レベル}dB)" 形式の文字列。
        """
        duration = (self._samples_seen - self._segment_start_sample) / SAMPLE_RATE
        return f"(音声 {duration:.1f}秒 / {level_db:.0f}dB)"

    def _make_event(self, is_final: bool) -> AsrEvent:
        """現在のセグメント状態からAsrEventを組み立てる.

        Args:
            is_final: 確定イベントかどうか。

        Returns:
            現在のセグメントを表すAsrEvent。
        """
        return AsrEvent(
            segment_id=self._segment_id,
            text=self._make_text(self._last_level_db),
            lang="ja",
            is_final=is_final,
            ts_audio_start=self._segment_start_sample / SAMPLE_RATE,
            ts_audio_end=self._samples_seen / SAMPLE_RATE,
        )

    def _process(self, pcm: np.ndarray) -> list[AsrEvent]:
        """1チャンク（約0.5秒）を処理する.

        Args:
            pcm: chunk_samples長のfloat32 PCM。

        Returns:
            発話中・進行中セグメントの無音継続中は暫定イベント1件、
            終端検出時は確定イベント1件、それ以外（発話開始前の無音）は空。

        副作用:
            processing_delay > 0 の場合、time.sleepで遅延を注入する
            （バックプレッシャ経路のテスト用）。
        """
        if self._processing_delay > 0.0:
            time.sleep(self._processing_delay)

        self._last_level_db = rms_dbfs(pcm)
        decision = self._endpointer.process(pcm)

        if decision == EndpointDecision.SPEECH and not self._segment_open:
            self._segment_open = True
            self._segment_start_sample = self._samples_seen
        self._samples_seen += len(pcm)

        if decision == EndpointDecision.SEGMENT_END:
            event = self._make_event(is_final=True)
            self._segment_id += 1
            self._segment_open = False
            return [event]

        if self._segment_open:
            return [self._make_event(is_final=False)]

        return []

    def _finalize(self) -> list[AsrEvent]:
        """flush/close時、進行中セグメントがあれば強制的に確定させる.

        Returns:
            進行中セグメントがあれば確定イベント1件、なければ空。
        """
        if not self._segment_open:
            return []
        self._endpointer.force_segment_end()
        event = self._make_event(is_final=True)
        self._segment_id += 1
        self._segment_open = False
        return [event]
