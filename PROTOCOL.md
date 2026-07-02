# tamami ストリーミング翻訳プロトコル仕様

- プロトコルバージョン: 1
- ステータス: ドラフト
- 作成日: 2026-07-02
- 改訂: 2026-07-02 トランスポートを WebSocket + raw PCM から WebRTC 構成へ変更
- 背景と設計判断は `STREAMING_PLAN.md` を参照

## 概要

tamami-remote-client（クライアント）と tamami（サーバー）間の通信仕様を定義する。
通信は 2 つのチャネルで構成する。

1. **WebSocket（制御チャネル）**: シグナリング（SDP 交換）・セッション制御・
   認識・翻訳結果の配信。すべて JSON テキストフレーム
2. **WebRTC（メディアチャネル）**: マイク音声の上り Opus オーディオトラック

```
[クライアント] --- WS: 制御 JSON（session_start / webrtc_offer 等）---> [tamami]
[クライアント] === WebRTC: Opus 音声トラック（上り）==================> [tamami]
[クライアント] <-- WS: 結果 JSON（asr / translation 等）--------------- [tamami]
```

## 用語

- **セッション**: WebSocket 接続 1 本と、それに紐づく RTCPeerConnection 1 つに対応する、
  1 マイク入力ストリームの処理単位
- **セグメント**: サーバーが発話区切り（VAD / ASR のエンドポイント検出）で定めた
  発話単位。認識・翻訳結果はセグメント単位で送られる
- **暫定 / 確定**: `is_final: false` のメッセージは同じセグメントの後続メッセージで
  置き換えられる。`is_final: true` で当該セグメントの当該種別（asr / translation）が
  確定し、以後更新されない

## 接続とハンドシェイク

1. クライアントは `ws://<host>:<port>/ws` へ接続する（ポートのデフォルトは 8765）
2. クライアントは最初のメッセージとして `session_start` を送る
3. サーバーは `session_ready` を返す。それ以前に `webrtc_offer` を送ってはならない
4. クライアントは RTCPeerConnection を作成し、マイクの音声トラックを追加して
   `webrtc_offer` を送る。サーバーは `webrtc_answer` を返す
5. ICE / DTLS の確立後、音声トラックの到着をもってサーバーは音声処理を開始する

補足:

- `protocol_version` が非対応の場合、サーバーは `error`（`unsupported_version`・
  `fatal: true`）を返して接続を閉じる
- ICE は non-trickle とする（候補収集の完了を待ち、候補を含んだ SDP を送る）。
  シグナリングのメッセージ数を減らし、実装を単純にするため
- バージョン 1 では STUN / TURN を使わない（ホスト候補のみ。LAN・同一 NAT 内を想定）。
  NAT 越えが必要な回線への対応は将来拡張とする

## 上り: 音声（WebRTC オーディオトラック）

- コーデックは Opus（WebRTC 標準・クロックレート 48kHz）。クライアントは SDP で
  Opus のみを提示する
- クライアントはマイク入力をオーディオトラックとして送出する。フレーム長は
  Opus 標準の 20ms。バッファを溜めず、取得し次第送出する
- サーバーは受信音声をデコードし、16kHz / mono へリサンプリングして ASR に渡す
- パケットの順序・欠落・ジッタは RTP とジッタバッファが処理するため、
  アプリケーション層でのシーケンス番号・タイムスタンプの付与は行わない
- **音声タイムライン**: サーバーが最初に受信した音声サンプルを 0 とした秒数を
  結果メッセージの `ts_audio_start` / `ts_audio_end` に用いる（受信サンプル数から算出）

## 上り: 制御メッセージ（JSON テキスト）

### session_start（必須・接続後最初に送る）

```json
{
  "type": "session_start",
  "protocol_version": 1,
  "client_ts_ms": 1751400000000
}
```

音声形式の宣言は行わない（コーデック・サンプルレートは SDP で交渉するため）。

### webrtc_offer（session_ready 受信後に送る）

```json
{ "type": "webrtc_offer", "sdp": "v=0\r\n..." }
```

### session_end（任意・正常終了時）

```json
{ "type": "session_end" }
```

クライアントはトラックの送出を止めてから送る。サーバーは処理中の音声をフラッシュし、
残りの確定結果をすべて送ってから接続を閉じる。クライアントが接続を切断した場合も
同様に扱うが、フラッシュ結果は届かない。

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

### webrtc_answer

```json
{ "type": "webrtc_answer", "sdp": "v=0\r\n..." }
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
  "server_ts_ms": 1751400037050
}
```

- `segment_id`: セッション内で単調増加する整数
- `lang`: 自動判定された言語（`"uk"` / `"ja"`）
- `ts_audio_start` / `ts_audio_end`: 音声タイムライン上の秒数（「上り: 音声」を参照）

### translation（訳文）

```json
{
  "type": "translation",
  "segment_id": 12,
  "text": "翻訳されたテキスト",
  "source_lang": "uk",
  "target_lang": "ja",
  "is_final": false,
  "ts_audio_start": 34.2,
  "ts_audio_end": 36.8,
  "server_ts_ms": 1751400037320
}
```

### pong

```json
{ "type": "pong", "client_ts_ms": 1751400000000, "server_ts_ms": 1751400000123 }
```

### error

```json
{ "type": "error", "code": "webrtc_failure", "message": "...", "fatal": true }
```

| code | 意味 | fatal |
|------|------|-------|
| `unsupported_version` | protocol_version が非対応 | true |
| `invalid_config` | session_start の内容が不正 | true |
| `webrtc_failure` | SDP 交渉・ICE / DTLS 確立・音声受信の失敗 | true |
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

- クライアントはトラック送出を開始した時刻 `t0_ms`（自分の時計）を記録する
- 体感遅延: 結果メッセージ受信時に
  `now_ms - (t0_ms + ts_audio_end * 1000)`
  を計算する。これは「その音声を話してから結果が表示されるまで」の実測値であり、
  `STREAMING_PLAN.md` の遅延目標（暫定表示 1 秒以内など）と直接比較できる
- サーバーの音声タイムライン原点（最初の受信サンプル）はクライアントの `t0_ms` より
  片道遅延 + ジッタバッファ分だけ遅れるため、体感遅延はその分だけ過小評価になる
  （LAN では数十 ms 程度であり、秒単位の目標判定には影響しない）
- サーバー内処理時間の内訳: `server_ts_ms` と ping / pong で得たオフセットの推定値を
  併用する。ただしこれは診断用であり、目標達成の判定には体感遅延を用いる
- RTT: `pong` 受信時の `now_ms - client_ts_ms`

## バージョニングと将来拡張

- 本仕様の非互換変更時は `protocol_version` を上げる。サーバーは対応しない
  バージョンを `unsupported_version` で拒否する
- 予約済みの拡張（バージョン 1 では未実装）:
  - STUN / TURN の利用（NAT 越えが必要な回線での運用時）
  - 結果配信の DataChannel 化（WebSocket を切り離す場合）
  - 下り TTS 音声のメディアトラック配信
  - `session_start` での翻訳方向・対象言語の明示指定（現在は uk⇄ja 自動判定のみ）
