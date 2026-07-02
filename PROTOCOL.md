# tamami ストリーミング翻訳プロトコル仕様

- プロトコルバージョン: 1
- ステータス: ドラフト
- 作成日: 2026-07-02
- 背景と設計判断は `STREAMING_PLAN.md` を参照

## 概要

tamami-remote-client（クライアント）と tamami（サーバー）間の通信仕様を定義する。
1 本の WebSocket 接続の上で、上りはバイナリフレーム（音声）、下りは JSON テキスト
フレーム（認識・翻訳結果）を流す。制御メッセージ（セッション開始・終了、ping）は
上りでも JSON テキストフレームを用いる。

```
[クライアント] --- バイナリ: 音声フレーム（ヘッダ + raw PCM）---> [tamami]
[クライアント] --- テキスト: 制御 JSON（session_start 等）------> [tamami]
[クライアント] <-- テキスト: 結果 JSON（asr / translation 等）--- [tamami]
```

## 用語

- **セッション**: WebSocket 接続 1 本に対応する、1 マイク入力ストリームの処理単位
- **セグメント**: サーバーが発話区切り（VAD / ASR のエンドポイント検出）で定めた
  発話単位。認識・翻訳結果はセグメント単位で送られる
- **暫定 / 確定**: `is_final: false` のメッセージは同じセグメントの後続メッセージで
  置き換えられる。`is_final: true` で当該セグメントの当該種別（asr / translation）が
  確定し、以後更新されない

## 接続とハンドシェイク

- エンドポイント: `ws://<host>:<port>/ws`（ポートのデフォルトは 8765）
- 接続確立後、クライアントは最初のメッセージとして `session_start` を送る
- サーバーは `session_ready` を返す。それ以前に音声フレームを送ってはならない
- `protocol_version` が非対応の場合、サーバーは `error`（`unsupported_version`・
  `fatal: true`）を返して接続を閉じる

## 上り: 音声バイナリフレーム

WebSocket バイナリフレーム 1 つが音声フレーム 1 つに対応する。

| オフセット | サイズ | 型 | 内容 |
|-----------|-------|-----|------|
| 0 | 4 | uint32 LE | シーケンス番号（0 始まり、フレームごとに +1） |
| 4 | 8 | uint64 LE | クライアント送信時刻（Unix epoch ミリ秒） |
| 12 | 可変 | bytes | 音声ペイロード（`session_start` で宣言した形式） |

- 音声形式は当面 raw PCM s16le / 16000Hz / mono のみ（`format: "pcm_s16le"`）
- 1 フレームのペイロードは 160〜320ms 分（16kHz で 2560〜5120 サンプル）を推奨。
  バッファを溜めず、取得し次第送信する
- シーケンス番号は欠落・順序の検証とデバッグに用いる（WebSocket は順序保証が
  あるため、通常は単調増加になる）
- 送信時刻はサーバーを経由して結果メッセージに反映され、遅延計測に用いる
  （「遅延計測」の節を参照）

## 上り: 制御メッセージ（JSON テキスト）

### session_start（必須・接続後最初に送る）

```json
{
  "type": "session_start",
  "protocol_version": 1,
  "audio": { "format": "pcm_s16le", "sample_rate": 16000, "channels": 1 },
  "client_ts_ms": 1751400000000
}
```

### session_end（任意・正常終了時）

```json
{ "type": "session_end" }
```

サーバーは処理中の音声をフラッシュし、残りの確定結果をすべて送ってから接続を閉じる。
クライアントが接続を切断した場合も同様に扱うが、フラッシュ結果は届かない。

### ping（任意・RTT 計測用）

```json
{ "type": "ping", "client_ts_ms": 1751400000000 }
```

## 下り: 結果メッセージ（JSON テキスト）

### session_ready

```json
{
  "type": "session_ready",
  "protocol_version": 1,
  "session_id": "a1b2c3d4",
  "server_ts_ms": 1751400000123
}
```

### asr（認識テキスト）

```json
{
  "type": "asr",
  "segment_id": 12,
  "text": "認識されたテキスト",
  "lang": "uk",
  "is_final": false,
  "ts_audio_start": 34.2,
  "ts_audio_end": 36.8,
  "last_audio_client_ts_ms": 1751400036800,
  "server_ts_ms": 1751400037050
}
```

- `segment_id`: セッション内で単調増加する整数
- `lang`: 自動判定された言語（`"uk"` / `"ja"`）
- `ts_audio_start` / `ts_audio_end`: セッション先頭を 0 とした音声タイムライン上の
  秒数（受信サンプル数から算出）
- `last_audio_client_ts_ms`: この結果の生成に使った最新の音声フレームの
  クライアント送信時刻（ヘッダの値をそのまま引き継ぐ）

### translation（訳文）

```json
{
  "type": "translation",
  "segment_id": 12,
  "text": "翻訳されたテキスト",
  "source_lang": "uk",
  "target_lang": "ja",
  "is_final": false,
  "last_audio_client_ts_ms": 1751400036800,
  "server_ts_ms": 1751400037320
}
```

### pong

```json
{ "type": "pong", "client_ts_ms": 1751400000000, "server_ts_ms": 1751400000123 }
```

### error

```json
{ "type": "error", "code": "audio_format_error", "message": "...", "fatal": true }
```

| code | 意味 | fatal |
|------|------|-------|
| `unsupported_version` | protocol_version が非対応 | true |
| `invalid_config` | session_start の内容が不正 | true |
| `audio_format_error` | 音声フレームの解釈に失敗 | true |
| `overloaded` | サーバー過負荷により受付不可 | true |
| `internal_error` | サーバー内部エラー | 場合による |

`fatal: true` の場合、サーバーは error 送信後に接続を閉じる。

## セグメントのライフサイクルと配信規則

1. サーバーは音声の到着に伴い、現在のセグメントについて `asr`（`is_final: false`）を
   繰り返し送る。テキストは毎回そのセグメントの全文であり、差分ではない
2. 再翻訳方式により、`asr` の更新に追随して `translation`（`is_final: false`）も
   繰り返し送られる。こちらも毎回全文
3. 発話区切りを検出すると `asr`（`is_final: true`）、続いて最終訳
   `translation`（`is_final: true`）を送り、セグメントを閉じる
4. 次の発話から新しい `segment_id` で 1. に戻る

配信規則:

- 同一 `segment_id`・同一 `type` のメッセージは、常に前の内容を**置き換える**
- `is_final: true` の後、同一 `segment_id`・同一 `type` のメッセージは送られない
- メッセージはセグメント内で順序どおりに届く（WebSocket の順序保証による）
- `asr` の確定と `translation` の確定は独立であり、`asr` が確定してから最終訳が
  届くまでに時間差がある

クライアント実装の指針（コンソール出力の例）:

- `is_final: false` → 現在行をキャリッジリターンで書き換える
- `is_final: true` → 内容を出力して改行し、次のセグメントに備える
- TTS など確定情報のみ扱う出力は `is_final: true` だけを消費すればよい

## 遅延計測

クロック同期を前提にせず、**クライアントは自分の時計だけで**エンドツーエンド遅延を
計測できるようにする。

- 体感遅延: 結果メッセージ受信時に
  `now_ms - last_audio_client_ts_ms`
  を計算する。これは「その音声を送ってから結果が表示されるまで」の実測値であり、
  `STREAMING_PLAN.md` の遅延目標（暫定表示 1 秒以内など）と直接比較できる
- サーバー内処理時間の内訳: `server_ts_ms` と ping/pong で得たオフセットの推定値を
  併用する。ただしこれは診断用であり、目標達成の判定には体感遅延を用いる
- RTT: `pong` 受信時の `now_ms - client_ts_ms`

## バージョニングと将来拡張

- 本仕様の非互換変更時は `protocol_version` を上げる。サーバーは対応しない
  バージョンを `unsupported_version` で拒否する
- 予約済みの拡張（バージョン 1 では未実装）:
  - 音声形式 `"opus"`（バイナリフレーム 1 つ = Opus パケット 1 つ。ヘッダは共通）
  - `session_start` での翻訳方向・対象言語の明示指定（現在は uk⇄ja 自動判定のみ）
  - 下り `tts_audio`（合成音声のバイナリ配信）
