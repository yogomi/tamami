"""NVIDIA Nemotron 3.5 ASR（Transformersストリーミング推論）認識器.

`nvidia/nemotron-3.5-asr-streaming-0.6b` をTransformers（>=5.13）の
`AutoModelForRNNT` ストリーミングAPIで動かすStreamingRecognizer実装。
Phase 0 実機検証（DGX_SETUP.md 0章）の結果に基づき、NeMo cache-aware APIの
旧ドラフトを置き換えてTransformersリリース版を採用した。

TransformersのストリーミングAPIは「特徴量のジェネレータをmodel.generateが
消費し、テキストをstreamerへ流す」pull型であり、push型のStreamingRecognizer
フック（_process / _finalize）とは制御の向きが逆になる。本実装ではセグメント
ごとに_GenerationSession（generate実行スレッド + 追記型音声バッファ）を立て、
_processは「バッファへの追記 → 追記分の処理完了待ち → テキスト回収」を行う
ことで両者を橋渡しする。

`transformers` / `torch` のインポートはモジュールのトップレベルに置かず、
モデルロード・セッション生成に限定している。これにより、未インストールの
Mac上でも（--asr fake を使う限り）サーバー本体は問題なく起動できる。
依存の導入は `uv sync --extra asr`（DGX Sparkではtorchがcu130 indexに解決
される。pyproject.tomlの[tool.uv.sources]を参照）。
"""

import logging
import queue
import re
import threading
import time
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

# チャンク長(ms) -> lookaheadトークン数のマッピング。
# processor.set_num_lookahead_tokens()がチャンク長（=公称ストリーミング遅延）を
# 決める（lookahead 0/3/6/13 = 80/320/560/1120ms。DGX_SETUP.md 0章の実測を参照。
# 旧NeMoドラフトにあった160msはTransformers経路には存在しない）。
CHUNK_MS_TO_LOOKAHEAD: dict[int, int] = {80: 0, 320: 3, 560: 6, 1120: 13}

# 確定テキスト末尾に付与される言語タグの形式（例: "<uk-UA>"）。
# タグからlocale先頭2文字（"uk"/"ja"等）を取り出す。
_LANG_TAG_RE = re.compile(r"<([a-z]{2})-[A-Z]{2}>\s*$")

# タグが得られるまでの初期言語（タグは通常セグメント末尾でしか出ない）。
_DEFAULT_LANG = "ja"

# 文字種による言語判定用（言語タグが出力されないセグメントのフォールバック。
# 実機検証でタグの出力は音声により不安定なことを確認済み）。
_CYRILLIC_RE = re.compile("[Ѐ-ӿ]")  # キリル文字
_JAPANESE_RE = re.compile("[぀-ヿ一-鿿]")  # ひらがな・カタカナ・CJK漢字


def detect_lang_from_script(text: str) -> Optional[str]:
    """文字種から言語を推定する（uk / ja の2言語前提のフォールバック）.

    キリル文字が多ければ "uk"、かな・漢字が多ければ "ja" を返す。
    どちらとも判定できない場合（数字・ラテン文字のみ等）はNoneを返す。

    Args:
        text: 判定対象のテキスト。

    Returns:
        "uk" / "ja" / None のいずれか。
    """
    cyrillic = len(_CYRILLIC_RE.findall(text))
    japanese = len(_JAPANESE_RE.findall(text))
    if cyrillic > japanese:
        return "uk"
    if japanese > cyrillic:
        return "ja"
    return None


# チャンクごとに付与する言語プロンプト（"auto"で自動判定させ、タグを受け取る）。
_LANGUAGE_PROMPT = "auto"

# feed済み音声の処理完了（テキスト回収可能）を待つ上限秒。超えても失敗にはせず、
# その時点までのテキストで暫定イベントを作る（次チャンクで追いつく）。
_PROCESSED_WAIT_TIMEOUT_SEC = 5.0

# セグメント確定時にgenerateスレッドの終了を待つ上限秒。
_JOIN_TIMEOUT_SEC = 30.0


def _load_model_and_processor() -> tuple[Any, Any]:
    """transformersを遅延importしてモデルと前処理器をロードする.

    --asr nemotron 指定時（NemotronStreamingRecognizer.load_model経由）にのみ
    呼ばれる。transformers未インストール環境でこのモジュールをimportしただけ
    では呼ばれないため、サーバー本体は壊れない。

    Returns:
        (モデル, プロセッサ) のタプル。CUDAが利用可能ならモデルはGPUに載せる。

    Raises:
        RuntimeError: transformersがインストールされていない場合
            （`uv sync --extra asr` で導入するよう促す）。
    """
    try:
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor
    except ImportError as e:
        raise RuntimeError(
            "transformers がインストールされていません。"
            "nemotron ASRの実行には `uv sync --extra asr` で "
            "asr extra（torch / transformers）を導入してください。"
        ) from e
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForRNNT.from_pretrained(MODEL_NAME)
    if torch.cuda.is_available():
        model = model.to("cuda")
    else:
        logger.warning("CUDAが利用できないためCPUで実行します（リアルタイム性能は保証されない）")
    model.eval()
    return model, processor


class _GenerationSession:
    """1セグメント分のmodel.generate実行を包む内部クラス.

    認識器のワーカースレッドから操作する前提。音声バッファへの追記をトリガに、
    専用スレッド上のgenerateが特徴量ジェネレータ経由でチャンクを消費し、
    テキストをstreamerへ流す。generateスレッドはこのセッションごとに1本で、
    finish()の呼び出しで必ず終了する（feederジェネレータはgenerate内部で
    動くため、別スレッドは増えない）。

    Args:
        model: ロード済みのモデル（Nemotron3_5AsrForRNNT）。
        processor: 対応するプロセッサ（Nemotron3_5AsrProcessor）。
    """

    def __init__(self, model: Any, processor: Any) -> None:
        """セッションを初期化する（generateスレッドはまだ起動しない）.

        Args:
            model: ロード済みのモデル。
            processor: 対応するプロセッサ。
        """
        from transformers import TextIteratorStreamer

        self._model = model
        self._processor = processor
        # 言語タグ（<xx-XX>）を受け取るためspecial tokenを残してデコードする
        self._streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=False)
        self._cond = threading.Condition()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._ended = False
        # feederが「ここまで処理し終えて待機に入った」バッファ位置（サンプル数）
        self._idle_at = -1
        self._thread: Optional[threading.Thread] = None
        self._failure: Optional[BaseException] = None
        self._text = ""

    def feed(self, pcm: np.ndarray) -> int:
        """音声をバッファへ追記する.

        最初のチャンク分が揃った時点でgenerateスレッドを起動する。

        Args:
            pcm: float32 PCM（16kHz / mono / [-1.0, 1.0]）。

        Returns:
            追記後のバッファ総サンプル数（wait_processedの引数に使う）。
        """
        with self._cond:
            self._buffer = np.concatenate([self._buffer, pcm])
            total = len(self._buffer)
            self._cond.notify_all()
        if self._thread is None and total >= self._processor.num_samples_first_audio_chunk:
            self._start_generate()
        return total

    def wait_processed(self, fed_samples: int, timeout: float) -> None:
        """feed済み音声の処理が一巡する（feederが待機に戻る）まで待つ.

        generateスレッドが未起動（最初のチャンク分が溜まっていない）・終了済み・
        失敗済みの場合は直ちに返る。タイムアウトしても例外にはしない
        （その時点までのテキストで暫定イベントを作れば、次チャンクで追いつく）。

        Args:
            fed_samples: feed()が返したバッファ総サンプル数。
            timeout: 待機の上限秒。
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            while (
                self._thread is not None
                and self._thread.is_alive()
                and self._failure is None
                and self._idle_at < fed_samples
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._cond.wait(remaining)

    def drain_text(self) -> str:
        """streamerに溜まったテキスト片を回収し、累積テキストを返す.

        Returns:
            セッション開始からの累積テキスト（言語タグ等のspecial tokenを含む）。
        """
        while True:
            try:
                piece = self._streamer.text_queue.get_nowait()
            except queue.Empty:
                break
            if piece == self._streamer.stop_signal:
                break
            self._text += piece
        return self._text

    def finish(self) -> str:
        """入力終了を指示し、generateの完了を待って最終テキストを返す.

        バッファ残余（1チャンク未満の端数）はゼロ詰めした最終チャンクとして
        処理される。generateスレッドが起動していなければ即座に返る。

        Returns:
            セグメント全体の確定テキスト（言語タグを含む）。
        """
        with self._cond:
            self._ended = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(_JOIN_TIMEOUT_SEC)
            if self._thread.is_alive():
                logger.warning("Nemotron generate thread did not finish within timeout")
        return self.drain_text()

    def raise_if_failed(self) -> None:
        """generateスレッドで例外が起きていた場合、それを再送出する.

        Raises:
            Exception: generateスレッドで発生した例外（CUDA OOM等）。
        """
        if self._failure is not None:
            raise self._failure

    def _start_generate(self) -> None:
        """最初のチャンクを組み立て、generateスレッドを起動する."""
        first_pcm = np.asarray(
            self._buffer[: self._processor.num_samples_first_audio_chunk], dtype=np.float32
        )
        first_inputs = self._processor(
            first_pcm,
            sampling_rate=SAMPLE_RATE,
            is_streaming=True,
            is_first_audio_chunk=True,
            language=_LANGUAGE_PROMPT,
            return_tensors="pt",
        ).to(self._model.device, dtype=self._model.dtype)
        kwargs = {
            **first_inputs,
            "input_features": self._feature_generator(first_inputs),
            "streamer": self._streamer,
        }
        self._thread = threading.Thread(
            target=self._run_generate, kwargs=kwargs, name="asr-generate", daemon=True
        )
        self._thread.start()

    def _run_generate(self, **kwargs: Any) -> None:
        """generate本体（専用スレッド）. 失敗はfailureに記録して待機側を起こす.

        Args:
            **kwargs: model.generateへ渡す引数一式。
        """
        try:
            self._model.generate(**kwargs)
        except Exception as exc:  # スレッド境界で捕捉し、raise_if_failed経由で伝搬する
            logger.exception("Nemotron generate thread failed")
            with self._cond:
                self._failure = exc
                self._cond.notify_all()

    def _feature_generator(self, first_inputs: Any) -> Any:
        """後続チャンクの特徴量を生成するジェネレータ（generateスレッド上で動く）.

        バッファに次のチャンク分が溜まるまで待機し、終了指示後は残余をゼロ詰め
        した最終チャンクを出して終わる。チャンクの切り出し位置はメルフレーム
        位置から逆算する（チャンク間でn_fft/2ぶん重なる。poc/streaming_latency.py
        と同じ計算）。

        Args:
            first_inputs: 最初のチャンクのprocessor出力。

        Yields:
            各チャンクのinput_featuresテンソル。
        """
        proc = self._processor
        yield first_inputs.input_features[:, : proc.num_mel_frames_first_audio_chunk, :]

        mel_idx = proc.num_mel_frames_first_audio_chunk
        hop = proc.feature_extractor.hop_length
        n_fft = proc.feature_extractor.n_fft
        chunk = proc.num_samples_per_audio_chunk
        while True:
            start = mel_idx * hop - n_fft // 2
            end = start + chunk
            with self._cond:
                # yieldから戻ってここに来た時点で、前チャンクのデコードは完了している
                while len(self._buffer) < end and not self._ended:
                    self._idle_at = len(self._buffer)
                    self._cond.notify_all()
                    self._cond.wait()
                if len(self._buffer) < end:
                    # 終了指示済み: 残余をゼロ詰めした最終チャンクを作る
                    if len(self._buffer) <= start:
                        return
                    pcm = np.zeros(chunk, dtype=np.float32)
                    tail = self._buffer[start:]
                    pcm[: len(tail)] = tail
                    is_last = True
                else:
                    pcm = np.asarray(self._buffer[start:end], dtype=np.float32)
                    is_last = False
            inputs = proc(
                pcm,
                sampling_rate=SAMPLE_RATE,
                is_streaming=True,
                is_first_audio_chunk=False,
                language=_LANGUAGE_PROMPT,
                return_tensors="pt",
            ).to(self._model.device, dtype=self._model.dtype)
            yield inputs.input_features
            if is_last:
                return
            mel_idx += proc.num_mel_frames_per_audio_chunk


class NemotronStreamingRecognizer(StreamingRecognizer):
    """Nemotron 3.5 ASR（Transformersストリーミング推論）によるストリーミング認識器.

    プロセス内でモデルを1回だけロードし（load_model classmethod）、以後の
    インスタンスは共有モデル + per-streamの_GenerationSessionのみを持つ
    （複数セッションで600Mモデルの重みを共有し、メモリを節約するため）。
    推論はセッションごとのgenerateスレッド上で走る。PyTorchのCUDA呼び出しは
    スレッドセーフなためロックは置かない（複数セッション同時実行時の
    スループット最適化は、推論サーバー分離を検討する段階で扱う）。

    Args:
        chunk_ms: 1チャンクあたりの音声長（ミリ秒）。80/320/560/1120のいずれかで、
            load_model()と同じ値を指定すること（lookahead設定がprocessorに対して
            プロセス全体で1つのため）。
        max_buffered_seconds: 基底クラス参照（バックプレッシャ検出の閾値）。

    Raises:
        RuntimeError: load_model()が未実行の場合。
        ValueError: chunk_msが対応表にない、またはload_model()時と異なる場合。
    """

    # プロセス内で共有するモデル・プロセッサ（load_model()で1回だけセットする）。
    _model: ClassVar[Optional[Any]] = None
    _processor: ClassVar[Optional[Any]] = None
    _loaded_chunk_ms: ClassVar[Optional[int]] = None
    _model_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def load_model(cls, chunk_ms: int) -> None:
        """プロセス内で1回だけモデルをロードし、クラス変数に保持する.

        サーバー起動時（src/server/__main__.py）に1回だけ呼ぶ想定。2回目以降は
        何もしない（同一プロセス内での再起動・複数回呼び出しに対して安全）。

        Args:
            chunk_ms: lookaheadトークン数の決定に使うチャンク長（ミリ秒）。

        Raises:
            ValueError: chunk_msが対応表にない場合。
            RuntimeError: transformersがインストールされていない場合
                （_load_model_and_processor参照）。
        """
        if chunk_ms not in CHUNK_MS_TO_LOOKAHEAD:
            raise ValueError(
                f"unsupported chunk_ms: {chunk_ms} (choices: {sorted(CHUNK_MS_TO_LOOKAHEAD)})"
            )
        with cls._model_lock:
            if cls._model is not None:
                logger.info("Nemotron ASR model already loaded, skipping reload")
                return
            logger.info("loading Nemotron ASR model: %s", MODEL_NAME)
            model, processor = _load_model_and_processor()
            processor.set_num_lookahead_tokens(CHUNK_MS_TO_LOOKAHEAD[chunk_ms])
            cls._model = model
            cls._processor = processor
            cls._loaded_chunk_ms = chunk_ms
            logger.info(
                "Nemotron ASR model loaded (chunk_ms=%d, nominal latency=%dms, samples/chunk=%d)",
                chunk_ms,
                processor.streaming_latency_ms,
                processor.num_samples_per_audio_chunk,
            )

    def __init__(self, chunk_ms: int = 560, max_buffered_seconds: float = 10.0) -> None:
        """認識器を初期化する（per-streamのセッション状態を持つ）.

        Args:
            chunk_ms: 1チャンクあたりの音声長（ミリ秒）。load_model()と同じ値を
                指定すること。
            max_buffered_seconds: バックプレッシャ検出の閾値（秒）。

        Raises:
            RuntimeError: load_model()が未実行の場合。
            ValueError: chunk_msが対応表にない、またはload_model()時と異なる場合。
        """
        cls = NemotronStreamingRecognizer
        if cls._model is None or cls._processor is None:
            raise RuntimeError(
                "NemotronStreamingRecognizer.load_model() が呼ばれていません。"
                "サーバー起動時に一度だけロードしてください。"
            )
        if chunk_ms not in CHUNK_MS_TO_LOOKAHEAD:
            raise ValueError(
                f"unsupported chunk_ms: {chunk_ms} (choices: {sorted(CHUNK_MS_TO_LOOKAHEAD)})"
            )
        if chunk_ms != cls._loaded_chunk_ms:
            raise ValueError(
                f"chunk_ms mismatch: load_model({cls._loaded_chunk_ms}) と異なる値 "
                f"{chunk_ms} が指定された"
            )
        # 基底クラスの再バッファは「モデルが要求する1チャンクのサンプル数」に合わせる
        # （processorのlookahead設定に依存し、chunk_ms/1000*SAMPLE_RATEとは一致しない）。
        super().__init__(
            max_buffered_seconds=max_buffered_seconds,
            chunk_samples=int(cls._processor.num_samples_per_audio_chunk),
        )
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
        self._session: Optional[_GenerationSession] = None

    def _extract_lang(self, text: str) -> tuple[str, str]:
        """テキスト末尾の言語タグ（例: "<uk-UA>"）を取り除き、言語コードを返す.

        タグが見つからない場合は文字種による推定（detect_lang_from_script）に
        フォールバックする（タグの出力は音声により不安定なため）。それでも
        判定できない場合は前回値を保持する（初期値は"ja"）。

        Args:
            text: モデルが返した生テキスト。

        Returns:
            (タグを除いたテキスト, 言語コード) のタプル。
        """
        match = _LANG_TAG_RE.search(text)
        if match:
            self._current_lang = match.group(1)
            return _LANG_TAG_RE.sub("", text).rstrip(), self._current_lang
        detected = detect_lang_from_script(text)
        if detected is not None:
            self._current_lang = detected
        return text, self._current_lang

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

    def _close_segment(self) -> list[AsrEvent]:
        """現在のセッションを終了し、進行中セグメントがあれば確定イベントを返す.

        Returns:
            進行中セグメントがあれば確定イベント1件、なければ空。

        Raises:
            Exception: generateスレッドで発生していた例外。
        """
        session = self._session
        self._session = None
        raw_final = ""
        if session is not None:
            raw_final = session.finish()
            session.raise_if_failed()
        if not self._segment_open:
            return []
        if raw_final:
            text, lang = self._extract_lang(raw_final)
        else:
            text, lang = self._last_text, self._current_lang
        event = self._make_event(text, lang, is_final=True)
        self._segment_id += 1
        self._segment_open = False
        self._last_text = ""
        return [event]

    def _process(self, pcm: np.ndarray) -> list[AsrEvent]:
        """1チャンク分の音声をTransformersストリーミング推論で処理する.

        Args:
            pcm: chunk_samples長のfloat32 PCM（16kHz/mono/[-1.0, 1.0]）。

        Returns:
            発話区間中は暫定イベント、SilenceEndpointerが終端を検出した場合は
            確定イベントを含むリスト。

        Raises:
            Exception: generateスレッドで発生した例外（CUDA OOM等）。
        """
        decision = self._endpointer.process(pcm)
        if self._session is None:
            self._session = _GenerationSession(
                NemotronStreamingRecognizer._model, NemotronStreamingRecognizer._processor
            )
        session = self._session

        fed = session.feed(pcm)
        session.wait_processed(fed, timeout=_PROCESSED_WAIT_TIMEOUT_SEC)
        session.raise_if_failed()

        text, lang = self._extract_lang(session.drain_text())
        if not self._segment_open and text:
            self._segment_open = True
            self._segment_start_sample = self._samples_seen
        self._samples_seen += len(pcm)
        self._last_text = text

        if decision == EndpointDecision.SEGMENT_END:
            # セグメント確定。テキストが出ていない場合もセッションは作り直す
            # （次セグメントを is_first_audio_chunk=True から始めるため）。
            return self._close_segment()

        if self._segment_open:
            return [self._make_event(text, lang, is_final=False)]
        return []

    def _finalize(self) -> list[AsrEvent]:
        """flush/close時、進行中セグメントがあれば強制的に確定させる.

        Returns:
            進行中セグメントがあれば確定イベント1件、なければ空。

        副作用:
            進行中のセッション（generateスレッド）を終了させる。
        """
        self._endpointer.force_segment_end()
        return self._close_segment()
