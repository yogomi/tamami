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

## 0. 素の環境（コンテナなし）での事前検証の記録（2026-07-15）

コンテナを立ち上げる前に、DGX Spark の素の環境（uv venv）で GPU を使った検証を先に行い、
動作確認が取れてからコンテナ化する方針に変更した（検証の反復速度を優先する。
コンテナ化は 1 章以降の手順で後追いする）。

### 検証結果の要約

| 項目 | 結果 |
|------|------|
| Python 3.13.13 venv（uv、aarch64） | OK |
| torch 2.13.0+cu130（cu130 index） | OK。GB10 認識（compute capability 12.1）、GPU 行列積 OK |
| nemo_toolkit 2.7.3（PyPI 最新） | インストール可。ただし本モデルのロードは不可（下記 1） |
| NeMo git main（3.1.0+24c4f58a7） | OK。モデルロード・GPU 推論とも成功 |
| Nemotron 3.5 ASR の GPU 推論 | OK。`transcribe()` スモークテスト成功（正弦波入力 → 空文字） |

### 判明した事実

1. **NeMo は git main が必須**。モデルの target クラス
   `nemo.collections.asr.models.rnnt_bpe_models_prompt.EncDecRNNTBPEModelWithPrompt`
   （LangID プロンプト対応 RNNT）は PyPI リリース版（2.7.3）に未収録。PyPI 版で
   `ASRModel.from_pretrained` を呼ぶと抽象クラスへフォールバックし
   `TypeError: Can't instantiate abstract class ASRModel ...` で失敗する。
   モデルカードも `git+https://github.com/NVIDIA/NeMo.git@main` からの導入を指示している。
2. **依存解決には `numba>=0.61` の明示が必要**。無指定だと numba 0.53.1 / llvmlite 0.36.0
   （2021 年版、Python 3.13 の wheel なし）へ解決され、ソースビルドで失敗する。
3. torch は PyTorch 公式の cu130 index に aarch64 wheel があり、GB10（sm_121）で動作する。
   SPEC.md 策定時に懸念した「x86 前提の wheel 問題」は torch 本体については解消済み。
4. モデル重みは `~/.cache/huggingface/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b`
   （snapshot `f3d33339`）にキャッシュ済み。コンテナ化時は 1 章の起動コマンドの
   マウントでそのまま再利用できる。
5. `transcribe()` の主な引数は `audio` / `batch_size` / `return_hypotheses` などで、
   `target_lang` は直接現れない（プロンプト条件付けの渡し方は 2 章の検証対象）。

### コンテナ選定への影響

NGC イメージの選定条件に「**同梱 NeMo が `EncDecRNNTBPEModelWithPrompt` を含むこと**」が
加わる（digest 固定より先に確認する）。確認方法:

```bash
docker run --rm nvcr.io/nvidia/nemo:<タグ> python -c \
  "import nemo.collections.asr.models.rnnt_bpe_models_prompt"
```

### 代替経路（未評価）

モデルカードには Transformers（`transformers>=5.13.0`）の `AutoModelForRNNT` +
`AutoProcessor` によるストリーミング推論経路も記載されている
（`set_num_lookahead_tokens` / チャンクごとの言語プロンプト指定）。NeMo git main への
依存を避け、リリース版ライブラリにバージョン固定できる可能性があるため、
2 章の検証と並行して比較する。

### 再現手順（素の環境）

```bash
uv venv --python 3.13 <venv>
VIRTUAL_ENV=<venv> uv pip install torch --index-url https://download.pytorch.org/whl/cu130
VIRTUAL_ENV=<venv> uv pip install 'numba>=0.61' \
  'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@24c4f58a7643'
```

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

注意: 依存は上記2つで足りる。`pyproject.toml` の依存（numpy / aiortc / aiohttp）の
うち numpy はコンテナに同梱されている（Whisper 系旧実装とその依存
`openai-whisper` / `pyaudio` / `scipy` は削除済み）。

## 2. 実機検証チェックリスト

`src/speech/nemotron.py` は NeMo の cache-aware ストリーミング API のシグネチャが
未確認のまま書いたドラフト。`# TODO(DGX検証)` を付けた以下の 5 箇所を、コンテナ内の
NeMo ソースおよび公式サンプルと突き合わせて確認・修正する。

参照すべきサンプル（コンテナ内。パスは `pip show nemo_toolkit` で確認）:

- `examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py`

チェックリスト（`src/speech/nemotron.py` 内の該当関数）:

- [x] `ASRModel.from_pretrained(MODEL_NAME)` の引数・戻り値の型（`_load_model_impl`）
      → 0 章で確認済み。`from_pretrained(model_name=...)` が
      `EncDecRNNTBPEModelWithPrompt` を返す（NeMo git main が必要）
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
