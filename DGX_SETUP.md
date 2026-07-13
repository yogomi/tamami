# DGX Spark セットアップ・実機検証手順

作成日: 2026-07-13

DGX Spark 上で NGC NeMo コンテナを整備し、Nemotron ASR の実機検証
（STREAMING_PLAN.md 着手順 4 の残り）と NLLB 接続（着手順 5）を進めるための手順書。
Mac 側での設計・実装の経緯は `SPEC.md`・`STREAMING_PLAN.md`・`PROTOCOL.md` を参照。

## 前提環境

- DGX Spark（NVIDIA GB10、Grace Blackwell superchip）
- CPU: ARM64（aarch64、Grace CPU）— x86 前提の wheel は使えない場合がある
- CUDA 13.0 / Driver 580.142 / 統一メモリ 128GB
- 使用モデル: `nvidia/nemotron-3.5-asr-streaming-0.6b`（`src/speech/nemotron.py` の
  `MODEL_NAME`）

## 1. NGC NeMo コンテナの準備

### イメージの選定と digest 固定

1. [NGC カタログの NeMo コンテナ](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)
   から最新タグを確認する
2. そのタグが `linux/arm64` に対応しているかをマニフェストで確認する:

   ```bash
   docker manifest inspect nvcr.io/nvidia/nemo:<タグ> | grep -A2 arm64
   ```

3. pull できない場合は NGC アカウントでログインする（ユーザー名は `$oauthtoken` 固定、
   パスワードは NGC の API キー）:

   ```bash
   docker login nvcr.io
   ```

4. pull 後に digest を確認し、**下の「確定した構成の記録」欄に記入する**。以降の起動は
   タグではなく digest（`@sha256:...`）で行う（再現性のため。README.md の方針を参照）:

   ```bash
   docker images --digests nvcr.io/nvidia/nemo
   ```

### コンテナの起動

リポジトリと Hugging Face キャッシュ（モデル重みの再ダウンロード防止）をマウントし、
サーバーのポートを公開して起動する:

```bash
docker run --rm -it \
  --gpus all \
  --shm-size=8g \
  -v ~/workspace/karinui/tamami:/workspace/tamami \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8765:8765 \
  nvcr.io/nvidia/nemo@sha256:<digest> \
  bash
```

### コンテナ内の追加依存

サーバー本体（`src/server/`）は `numpy` / `aiortc` / `aiohttp` に依存する。numpy と
PyTorch・NeMo はコンテナに同梱されているが、aiortc / aiohttp は入っていない想定のため、
**コンテナ同梱の pip で**追加する（uv は使わない。理由は README.md 参照）:

```bash
pip install aiortc aiohttp
python -c "import aiortc, aiohttp, nemo.collections.asr"  # 導入確認
```

注意: `pyproject.toml` の `openai-whisper` / `pyaudio` / `scipy` は旧実装・クライアント
側の依存であり、`--asr nemotron` でのサーバー実行には不要（インストールしない）。

## 2. 実機検証チェックリスト

`src/speech/nemotron.py` は NeMo の cache-aware ストリーミング API のシグネチャが
未確認のまま書いたドラフト。`# TODO(DGX検証)` を付けた以下の 5 箇所を、コンテナ内の
NeMo ソースおよび公式サンプルと突き合わせて確認・修正する。

参照すべきサンプル（コンテナ内。パスは `pip show nemo_toolkit` で確認）:

- `examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py`

チェックリスト（`src/speech/nemotron.py` 内の該当関数）:

- [ ] `ASRModel.from_pretrained(MODEL_NAME)` の引数・戻り値の型（`_load_model_impl`）
- [ ] `att_context_size` の設定方法 — `encoder.set_default_att_context_size(...)` 相当の
      呼び出し規約（`load_model`）
- [ ] `encoder.get_initial_cache_state` の引数（batch_size 等）と戻り値のアンパック順序
- [ ] `model.preprocessor` の呼び出しシグネチャと、torch テンソル化・デバイス転送の要否
- [ ] `conformer_stream_step` の引数名・戻り値の個数と順序

確認後の動作検証:

- [ ] `load_model(560)` の単体実行（モデルロードと att_context_size 設定の smoke test）
- [ ] 短い wav（ウクライナ語・日本語各 1 本）をチャンク分割して食わせ、テキストと
      `<uk-UA>` 形式の言語タグが取れることを確認（タグの解釈は実装済み）
- [ ] `uv run pytest` 相当のテストは Mac 側で維持する（コンテナ内での実行は必須でない。
      test_nemotron.py は NeMo なしでの import 安全性を検証するもの）

## 3. サーバーの起動と E2E 検証

コンテナ内でコンテナ同梱の python から起動する（`uv run` は不可。README.md 参照）:

```bash
cd /workspace/tamami
python -m src.server --asr nemotron --chunk-ms 560
```

クライアント（Mac 側の tamami-remote-client）から `ws://<DGX SparkのIP>:8765/ws` へ
接続し、以下を確認する:

- [ ] 暫定 asr（is_final: false）が発話から 1 秒以内に届く（STREAMING_PLAN.md の目標 1）
- [ ] 無音 0.5 秒で確定 asr（is_final: true）が届く（SilenceEndpointer の既定値）
- [ ] session_end で残りの確定結果が届いてから接続が閉じる（PROTOCOL.md の契約）

## 4. チューニング（STREAMING_PLAN.md 着手順 6 の前倒し分）

- [ ] `--chunk-ms` を 80〜1120 で振り、遅延と精度のトレードオフを実測する
      （PROTOCOL.md の遅延計測を使用）
- [ ] `SilenceEndpointer` の閾値（`silence_threshold_db=-40.0` / `min_silence_sec=0.5`）を
      実環境のマイク・背景ノイズでチューニングする

## 5. NLLB（着手順 5）の配置方針

決定事項（2026-07-13、設計判断の経緯は SPEC.md「未決事項」参照）:

- NLLB は **同じ NGC NeMo コンテナ・同じ tamami プロセス内**で動かす
- ASR と同じ「ワーカースレッド + キュー」パターンで翻訳段を実装する（再翻訳方式）
- Triton 等への推論サーバー分離は、複数セッションの同時通訳が必要になるまで見送る
- 必要ライブラリは transformers / SentencePiece。NeMo コンテナに同梱されている想定
  だが、コンテナ整備時に確認する:

  ```bash
  python -c "import transformers, sentencepiece"
  ```

- モデルサイズ（600M / 1.3B / 3.3B）は GB10 上で実測して選定する（統一メモリ 128GB
  のため、どれを選んでもメモリ制約はない）

## 確定した構成の記録（実機作業時に記入）

| 項目 | 値 |
|------|-----|
| NGC イメージタグ | （記入） |
| digest | `sha256:`（記入） |
| NeMo バージョン | （記入） |
| transformers / sentencepiece 同梱 | （記入） |
| 採用チャンク長 | （記入） |
