"""NVIDIA Nemotron 3.5 ASR（streaming）認識器 — 実機未検証ドラフト.

`nvidia/nemotron-3.5-asr-streaming-0.6b`（NeMoのcache-awareストリーミングAPI）を
用いたStreamingRecognizer実装。

**注意**: このモジュールはGPU必須であり、このMac（darwin / CPU環境）では動作確認
できない。DGX Spark（NGCコンテナ）上での実機検証を前提としたドラフトである。
NeMoの正確なAPIシグネチャが手元で確認できないため、NeMo mainブランチの
cache-awareストリーミング例
（`examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py`）
の構造に沿って誠実に書いているが、実際のAPIと差異がある可能性が高い。検証が必要な
箇所には `# TODO(DGX検証)` を付けている。

`nemo`（および前処理で必要なtorchテンソル化）のインポートはモジュールの
トップレベルに置かず、モデルロード関数内に限定している。これにより、NeMo未
インストールのMac上でも（--asr fake を使う限り）サーバー本体は問題なく起動できる。
"""

import logging
import re
import threading
from typing import Any, ClassVar, Optional

import numpy as np

from src.speech.streaming import (
    SAMPLE_RATE,
    AsrEvent,
    EndpointDecision,
    SilenceEndpointer,
    StreamingRecognizer,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "nvidia/nemotron-3.5-asr-streaming-0.6b"

# チャンク長(ms) -> att_context_size のマッピング（80msフレーム単位、SPEC.md参照）。
# NeMoのcache-aware FastConformerはatt_context_sizeでチャンク長を制御する。
CHUNK_MS_TO_ATT_CONTEXT_SIZE: dict[int, list[int]] = {
    80: [56, 0],
    160: [56, 1],
    320: [56, 3],
    560: [56, 6],
    1120: [56, 13],
}

# 確定テキスト末尾に付与される言語タグの形式（例: "<uk-UA>"）。
# モデルカードの記載に基づく。タグからlocale先頭2文字（"uk"/"ja"等）を取り出す。
_LANG_TAG_RE = re.compile(r"<([a-z]{2})-[A-Z]{2}>\s*$")

# タグが得られるまでの初期言語（モデルカードに準拠した既定値）。
_DEFAULT_LANG = "ja"


def _load_nemo_asr_model() -> Any:
    """NeMoのASRModelを遅延importしてロードする.

    --asr nemotron 指定時（NemotronStreamingRecognizer.load_model経由）にのみ
    呼ばれる。NeMo未インストール環境でこのモジュールをimportしただけでは
    呼ばれないため、サーバー本体は壊れない。

    Returns:
        ロード済みのNeMo ASRModelインスタンス。

    Raises:
        RuntimeError: NeMoがインストールされていない場合
            （NGCコンテナ内で実行するか、NeMoをインストールするよう促す）。
    """
    try:
        import nemo.collections.asr as nemo_asr  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "NeMo (nemo_toolkit[asr]) がインストールされていません。"
            "nemotron ASRの実行にはNVIDIA NGCコンテナ内で実行するか、"
            "NeMoをインストールしてください。"
        ) from e
    # TODO(DGX検証): from_pretrained の引数・戻り値の型を実機で確認する
    return nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)


class NemotronStreamingRecognizer(StreamingRecognizer):
    """Nemotron 3.5 ASR（cache-aware streaming）によるストリーミング認識器.

    プロセス内でモデルを1回だけロードし（load_model classmethod）、以後の
    インスタンスは共有モデル + per-stream のキャッシュ状態のみを持つ
    （複数セッションで600Mモデルの重みを共有し、メモリを節約するため）。

    Args:
        chunk_ms: 1チャンクあたりの音声長（ミリ秒）。
            80/160/320/560/1120のいずれか。load_model()呼び出し時と同じ値を
            指定すること（att_context_sizeが対応するチャンク長で固定されるため）。
        max_buffered_seconds: 基底クラス参照（バックプレッシャ検出の閾値）。

    Raises:
        RuntimeError: load_model()が未実行の場合。
        ValueError: chunk_msが対応表にない場合。
    """

    # プロセス内で共有するモデル本体（load_model()で1回だけセットする）。
    _model: ClassVar[Optional[Any]] = None
    _model_lock: ClassVar[threading.Lock] = threading.Lock()
    # 単一GPUのCUDAコンテキスト保護のため、推論呼び出しをクラスレベルで直列化する。
    # 複数インスタンスを並列に動かしてスケールさせる場合はこのロックが律速になるため、
    # 将来スループットが問題になった場合はストリームごとのバッチ化等の再設計が必要
    # （現状は正しさを優先し、単純な直列化にとどめる）。
    _stream_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def load_model(cls, chunk_ms: int) -> None:
        """プロセス内で1回だけモデルをロードし、クラス変数に保持する.

        サーバー起動時（src/server/__main__.py）に1回だけ呼ぶ想定。2回目以降は
        何もしない（同一プロセス内での再起動・複数回呼び出しに対して安全）。

        Args:
            chunk_ms: att_context_sizeの決定に使うチャンク長（ミリ秒）。

        Raises:
            ValueError: chunk_msが対応表にない場合。
            RuntimeError: NeMoがインストールされていない場合
                （_load_nemo_asr_model参照）。
        """
        if chunk_ms not in CHUNK_MS_TO_ATT_CONTEXT_SIZE:
            raise ValueError(
                f"unsupported chunk_ms: {chunk_ms} "
                f"(choices: {sorted(CHUNK_MS_TO_ATT_CONTEXT_SIZE)})"
            )
        with cls._model_lock:
            if cls._model is not None:
                logger.info("Nemotron ASR model already loaded, skipping reload")
                return
            logger.info("loading Nemotron ASR model: %s", MODEL_NAME)
            model = _load_nemo_asr_model()
            att_context_size = CHUNK_MS_TO_ATT_CONTEXT_SIZE[chunk_ms]
            # TODO(DGX検証): att_context_sizeの設定方法（メソッド名・呼び出し規約）を
            # 実機で確認する。cache-aware streaming例では
            # model.encoder.set_default_att_context_size(...) 相当の呼び出しを想定。
            model.encoder.set_default_att_context_size(att_context_size)
            model.eval()
            cls._model = model
            logger.info("Nemotron ASR model loaded (att_context_size=%s)", att_context_size)

    def __init__(self, chunk_ms: int = 560, max_buffered_seconds: float = 10.0) -> None:
        """認識器を初期化する（per-streamのキャッシュ状態を持つ）.

        Args:
            chunk_ms: 1チャンクあたりの音声長（ミリ秒）。load_model()と同じ値を
                指定すること。
            max_buffered_seconds: バックプレッシャ検出の閾値（秒）。

        Raises:
            RuntimeError: load_model()が未実行の場合。
            ValueError: chunk_msが対応表にない場合。
        """
        if NemotronStreamingRecognizer._model is None:
            raise RuntimeError(
                "NemotronStreamingRecognizer.load_model() が呼ばれていません。"
                "サーバー起動時に一度だけロードしてください。"
            )
        if chunk_ms not in CHUNK_MS_TO_ATT_CONTEXT_SIZE:
            raise ValueError(
                f"unsupported chunk_ms: {chunk_ms} "
                f"(choices: {sorted(CHUNK_MS_TO_ATT_CONTEXT_SIZE)})"
            )
        chunk_samples = int(chunk_ms / 1000 * SAMPLE_RATE)
        super().__init__(max_buffered_seconds=max_buffered_seconds, chunk_samples=chunk_samples)
        self._chunk_ms = chunk_ms
        # セグメント確定はモデルカードにEOU検出の記載がないため、
        # Nemotron側ではなくSilenceEndpointerで行う。
        self._endpointer = SilenceEndpointer()
        self._segment_id = 0
        self._segment_open = False
        self._segment_start_sample = 0
        self._samples_seen = 0
        self._current_lang = _DEFAULT_LANG
        self._last_text = ""
        self._cache: dict[str, Any] = {}
        self._previous_hypotheses: Optional[Any] = None
        self._reset_stream()

    def _reset_stream(self) -> None:
        """per-streamのキャッシュ状態とhypothesesを（再）初期化する.

        コンストラクタ・セグメント確定時（次セグメントへ移る際）に呼ぶ。
        """
        model = NemotronStreamingRecognizer._model
        # TODO(DGX検証): get_initial_cache_state の引数（batch_size等）・
        # 戻り値のアンパック順序を実機で確認する。
        (
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
        ) = model.encoder.get_initial_cache_state(batch_size=1)
        self._cache = {
            "cache_last_channel": cache_last_channel,
            "cache_last_time": cache_last_time,
            "cache_last_channel_len": cache_last_channel_len,
        }
        self._previous_hypotheses = None

    def _preprocess(self, pcm: np.ndarray) -> tuple[Any, Any]:
        """PCMをモデルの前処理器（特徴量抽出）にかける.

        TODO(DGX検証): 実際にはtorchテンソル化・model.preprocessorの呼び出し・
        デバイス転送（.cuda()等）が必要。cache-aware streaming例の
        `preprocess_audio`相当の処理を実機で確認して実装する。

        Args:
            pcm: chunk_samples長のfloat32 PCM（16kHz/mono/[-1.0, 1.0]）。

        Returns:
            (processed_signal, processed_signal_length) のタプル。
        """
        import torch

        model = NemotronStreamingRecognizer._model
        audio_signal = torch.from_numpy(pcm).unsqueeze(0)
        audio_signal_length = torch.tensor([len(pcm)])
        # TODO(DGX検証): model.preprocessor のシグネチャ（引数名・戻り値）を確認する
        processed_signal, processed_signal_length = model.preprocessor(
            input_signal=audio_signal, length=audio_signal_length
        )
        return processed_signal, processed_signal_length

    def _extract_lang(self, text: str) -> tuple[str, str]:
        """確定テキスト末尾の言語タグ（例: "<uk-UA>"）を取り除き、言語コードを返す.

        タグが見つからない場合（暫定テキストなど）は前回値を保持する
        （初期値は"ja"）。

        Args:
            text: モデルが返した生テキスト。

        Returns:
            (タグを除いたテキスト, 言語コード) のタプル。
        """
        match = _LANG_TAG_RE.search(text)
        if not match:
            return text, self._current_lang
        lang = match.group(1)
        stripped = _LANG_TAG_RE.sub("", text).rstrip()
        self._current_lang = lang
        return stripped, lang

    def _make_event(self, text: str, lang: str, is_final: bool) -> AsrEvent:
        """現在のセグメント状態からAsrEventを組み立てる.

        Args:
            text: セグメントの認識テキスト全文（言語タグ除去済み）。
            lang: 言語コード。
            is_final: 確定イベントかどうか。

        Returns:
            現在のセグメントを表すAsrEvent。
        """
        return AsrEvent(
            segment_id=self._segment_id,
            text=text,
            lang=lang,
            is_final=is_final,
            ts_audio_start=self._segment_start_sample / SAMPLE_RATE,
            ts_audio_end=self._samples_seen / SAMPLE_RATE,
        )

    def _process(self, pcm: np.ndarray) -> list[AsrEvent]:
        """1チャンク分の音声をNeMoのcache-awareストリーミングAPIで処理する.

        Args:
            pcm: chunk_samples長のfloat32 PCM（16kHz/mono/[-1.0, 1.0]）。

        Returns:
            暫定認識イベント（発話区間中）に加え、SilenceEndpointerが終端を
            検出した場合は確定イベントを含むリスト。

        副作用:
            推論本体（conformer_stream_step相当）はクラスレベルの
            _stream_lockで直列化する（単一GPUのCUDAコンテキスト保護）。
        """
        decision = self._endpointer.process(pcm)
        processed_signal, processed_signal_length = self._preprocess(pcm)

        model = NemotronStreamingRecognizer._model
        with NemotronStreamingRecognizer._stream_lock:
            # TODO(DGX検証): conformer_stream_step の引数名・戻り値の個数・順序を
            # 実機で確認する。ここではspeech_to_text_cache_aware_streaming_infer.py
            # 相当の呼び出し形を想定している。
            (
                transcribed_texts,
                self._cache["cache_last_channel"],
                self._cache["cache_last_time"],
                self._cache["cache_last_channel_len"],
                self._previous_hypotheses,
            ) = model.conformer_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=self._cache["cache_last_channel"],
                cache_last_time=self._cache["cache_last_time"],
                cache_last_channel_len=self._cache["cache_last_channel_len"],
                keep_all_outputs=False,
                previous_hypotheses=self._previous_hypotheses,
            )

        raw_text = transcribed_texts[0] if transcribed_texts else ""
        text, lang = self._extract_lang(raw_text)
        self._last_text = text

        if not self._segment_open and text:
            self._segment_open = True
            self._segment_start_sample = self._samples_seen
        self._samples_seen += len(pcm)

        events: list[AsrEvent] = []
        if self._segment_open:
            events.append(self._make_event(text, lang, is_final=False))

        if decision == EndpointDecision.SEGMENT_END and self._segment_open:
            events.append(self._make_event(text, lang, is_final=True))
            self._segment_id += 1
            self._segment_open = False
            self._reset_stream()

        return events

    def _finalize(self) -> list[AsrEvent]:
        """flush/close時、進行中セグメントがあれば強制的に確定させる.

        Returns:
            進行中セグメントがあれば確定イベント1件、なければ空。

        副作用:
            確定させた場合はキャッシュ状態・hypothesesをリセットする。
        """
        if not self._segment_open:
            return []
        self._endpointer.force_segment_end()
        event = self._make_event(self._last_text, self._current_lang, is_final=True)
        self._segment_id += 1
        self._segment_open = False
        self._reset_stream()
        return [event]
