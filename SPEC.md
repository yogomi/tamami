# tamami 仕様書

## 概要

ウクライナ語⇄日本語のリアルタイム通訳システム。マイクから取得した音声を認識（ASR）し、
翻訳して出力する。すべてローカル環境で動作させることを前提とする。

- 対応言語：ウクライナ語 ⇔ 日本語（双方向）
- 言語検出：自動（話者による言語指定なし）
- 話者識別：なし
- 許容遅延：可能な限りリアルタイムに近づける（旧仕様は 6〜12 秒。
  再策定した遅延目標と方針は `STREAMING_PLAN.md` を参照）
- 音声入力：クライアント（tamami-remote-client）のマイクから WebRTC で受信
  （サーバー側の受信処理は `src/server/session.py`）

---

## 音声認識（ASR）の採用方針

### 採用モデル：NVIDIA Nemotron 3.5 ASR（streaming, 0.6B）

音声認識には **`nvidia/nemotron-3.5-asr-streaming-0.6b`** を採用する。

初期の実装は OpenAI Whisper（`WhisperRecognizer`、`openai-whisper` パッケージ）を
用いていたが、Nemotron 3.5 ASR（`src/speech/nemotron.py`）へ移行済み。
Whisper 系の旧実装（マイク入力・バッチ認識のローカルパイプライン）は削除した。

### 採用理由

- **双方向を単一モデルで認識**：日本語（ja）・ウクライナ語（uk）の両方を 1 モデルで扱える。
  Whisper と異なり、認識側をモデル 1 つに統一できる。
- **ネイティブストリーミング**：Cache-Aware FastConformer + RNNT 構成により、
  チャンクを重複なく 1 回ずつ処理する真のストリーミングに対応。Whisper のような
  疑似ストリーミング（チャンク分割 + VAD）に伴う境界での精度低下・区切り待ち遅延を回避できる。
- **低遅延**：発話終端レイテンシは sub-100ms。許容遅延 6〜12 秒に対して十分な余裕がある。
- **自動言語検出**：プロンプトなしで言語を判定できる（LangID プロンプト条件付けも可能）。
- **商用利用可**：ライセンスは OpenMDW-1.1。商用利用が許可されている。

### 主要スペック

| 項目 | 内容 |
|------|------|
| パラメータ数 | 600M |
| 対応言語 | 40 言語ロケール（ja-JP・uk-UA を含む） |
| チャンクサイズ | 80ms / 160ms / 320ms / 560ms / 1120ms（可変） |
| 発話終端レイテンシ | sub-100ms |
| ライセンス | OpenMDW-1.1（商用利用可） |
| フレームワーク | NeMo（主） / Transformers、Python ≥3.11 + PyTorch |

### 精度（1.12s チャンク・LangID モード）

- 日本語（ja-JP）：CER 11.48%（日本語は WER ではなく CER で評価）
- ウクライナ語（uk-UA）：WER 13.07%
- ロシア語（ru-RU）：WER 9.17%

---

## 翻訳

Nemotron 3.5 ASR は **ASR 専用であり、翻訳機能を持たない**（音声 → 同一言語テキストまで）。
したがって翻訳は別モデルで行う 2 段構成とする。

```
音声 → [Nemotron 3.5 ASR] → テキスト（同一言語） → [翻訳モデル: NLLB] → 訳文
```

- 翻訳には **NLLB** を使用する。
- 認識テキストと訳文の両方を取り出せる 2 段構成とすることで、途中テキストの表示や
  各段の個別チューニングを可能にする。

---

## 実行環境

- **GPU**：NVIDIA GB10（DGX Spark、Grace Blackwell superchip）
  - アーキテクチャ：Blackwell（Nemotron の対応アーキテクチャに合致）
  - CUDA 13.0 / Driver 580.142
  - 統一メモリ 128GB（600M モデルには十分）
- **CPU アーキテクチャ**：ARM64（aarch64、Grace CPU）
  - x86 前提の wheel がそのまま使えない場合があるため、**NVIDIA 公式コンテナ（NGC）**での
    運用を推奨する（NeMo コンテナの ARM64 / Blackwell 対応イメージを使用）。
  - 素の pip で構築する場合は、NVIDIA 配布の aarch64 + CUDA 13 向け PyTorch ビルドを明示指定する。

---

## 未決事項

- ~~Whisper（`WhisperRecognizer`）から Nemotron 3.5 ASR への具体的な移行手順・
  インターフェース設計~~ → 解決済み。`src/speech/streaming.py` に
  `StreamingRecognizer` 抽象（ワーカースレッド・チャンク再バッファ・
  バックプレッシャ検出を共通化し、サブクラスは `_process` / `_finalize` のみ実装する）
  を導入し、開発・テスト用の `FakeStreamingRecognizer`（`src/speech/fake.py`）と
  Nemotronドラフト実装 `NemotronStreamingRecognizer`（`src/speech/nemotron.py`）を
  用意した。`src/server/session.py` はこの抽象経由でASRに接続する構成になっている。
- **Nemotron 3.5 ASR の実機検証（DGX Spark / NGCコンテナ）**：
  `src/speech/nemotron.py` はNeMoのcache-awareストリーミングAPI
  （`get_initial_cache_state` / `conformer_stream_step` 等）の正確なシグネチャが
  手元で確認できないまま書いたドラフトであり、`# TODO(DGX検証)` を付けた箇所を
  実機で確認・修正する必要がある。
- チャンクサイズ（80ms〜1120ms、`att_context_size` 経由）と精度・遅延のトレードオフ検証
  （実機検証と合わせて行う）。
- **EOU代替の無音endpointingのチューニング**：セグメント確定は現状
  `SilenceEndpointer`（`silence_threshold_db` / `min_silence_sec`）で行っている
  （モデルカードにEOU検出の記載がないため）。閾値・継続秒数の実環境（マイク・
  背景ノイズ）でのチューニングが必要。将来的にはNemotron側の実発話終端検出
  （提供されれば）やSilero VAD等への差し替えも検討する。
- ASR 出力テキストと NLLB 翻訳段の連携方法（逐次翻訳の単位・タイミング）
