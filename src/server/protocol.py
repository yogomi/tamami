"""ストリーミング翻訳プロトコル（バージョン 1）のメッセージ定義.

PROTOCOL.md に定義されたワイヤープロトコルのメッセージ生成・検証を行う。
メッセージはすべて dict として扱い、送受信側で JSON との変換を行う。
"""

import json
import time
from typing import Any

PROTOCOL_VERSION = 1


class ProtocolError(Exception):
    """プロトコル違反を表す例外.

    Attributes:
        code: PROTOCOL.md の error メッセージに定義されたエラーコード。
        message: 人間可読なエラー内容。
        fatal: True の場合、error 送信後に接続を閉じるべきであることを示す。
    """

    def __init__(self, code: str, message: str, fatal: bool = True) -> None:
        """例外を初期化する.

        Args:
            code: エラーコード（例: "invalid_config"）。
            message: エラー内容の説明。
            fatal: 接続を閉じるべきかどうか。
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.fatal = fatal


def now_ms() -> int:
    """現在時刻を Unix epoch ミリ秒で返す.

    Returns:
        Unix epoch からの経過ミリ秒。
    """
    return int(time.time() * 1000)


def parse_client_message(raw: str) -> dict[str, Any]:
    """クライアントからの JSON テキストフレームを解析する.

    Args:
        raw: 受信した JSON 文字列。

    Returns:
        解析済みメッセージ。"type" キーの存在は保証される。

    Raises:
        ProtocolError: JSON として不正、または "type" が文字列でない場合
            （code: "invalid_config"）。
    """
    try:
        message = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ProtocolError("invalid_config", f"invalid JSON: {e}") from e
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise ProtocolError(
            "invalid_config", "message must be an object with a string 'type'"
        )
    return message


def validate_session_start(message: dict[str, Any]) -> None:
    """session_start メッセージを検証する.

    Args:
        message: parse_client_message で解析済みのメッセージ。

    Raises:
        ProtocolError: protocol_version が非対応の場合（code: "unsupported_version"）、
            または必須フィールドが不正な場合（code: "invalid_config"）。
    """
    version = message.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            "unsupported_version",
            f"protocol_version {version!r} is not supported (server: {PROTOCOL_VERSION})",
        )
    if not isinstance(message.get("client_ts_ms"), int):
        raise ProtocolError("invalid_config", "client_ts_ms must be an integer")


def make_session_ready(session_id: str) -> dict[str, Any]:
    """session_ready メッセージを生成する.

    Args:
        session_id: セッション識別子。

    Returns:
        session_ready メッセージ。
    """
    return {
        "type": "session_ready",
        "protocol_version": PROTOCOL_VERSION,
        "session_id": session_id,
        "server_ts_ms": now_ms(),
    }


def make_webrtc_answer(sdp: str) -> dict[str, Any]:
    """webrtc_answer メッセージを生成する.

    Args:
        sdp: ICE 候補を含む answer の SDP。

    Returns:
        webrtc_answer メッセージ。
    """
    return {"type": "webrtc_answer", "sdp": sdp}


def make_asr(
    segment_id: int,
    text: str,
    lang: str,
    is_final: bool,
    ts_audio_start: float,
    ts_audio_end: float,
) -> dict[str, Any]:
    """asr（認識テキスト）メッセージを生成する.

    Args:
        segment_id: セッション内で単調増加するセグメント番号。
        text: セグメントの認識テキスト全文（差分ではない）。
        lang: 判定された言語コード（"uk" / "ja"）。
        is_final: セグメントの認識が確定したかどうか。
        ts_audio_start: 音声タイムライン上のセグメント開始秒。
        ts_audio_end: 音声タイムライン上のセグメント終了秒。

    Returns:
        asr メッセージ。
    """
    return {
        "type": "asr",
        "segment_id": segment_id,
        "text": text,
        "lang": lang,
        "is_final": is_final,
        "ts_audio_start": ts_audio_start,
        "ts_audio_end": ts_audio_end,
        "server_ts_ms": now_ms(),
    }


def make_translation(
    segment_id: int,
    text: str,
    source_lang: str,
    target_lang: str,
    is_final: bool,
    ts_audio_start: float,
    ts_audio_end: float,
) -> dict[str, Any]:
    """translation（訳文）メッセージを生成する.

    Args:
        segment_id: 対応する asr と同じセグメント番号。
        text: セグメントの訳文全文（差分ではない）。
        source_lang: 原文の言語コード。
        target_lang: 訳文の言語コード。
        is_final: セグメントの訳文が確定したかどうか。
        ts_audio_start: 音声タイムライン上のセグメント開始秒。
        ts_audio_end: 音声タイムライン上のセグメント終了秒。

    Returns:
        translation メッセージ。
    """
    return {
        "type": "translation",
        "segment_id": segment_id,
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "is_final": is_final,
        "ts_audio_start": ts_audio_start,
        "ts_audio_end": ts_audio_end,
        "server_ts_ms": now_ms(),
    }


def make_pong(client_ts_ms: int) -> dict[str, Any]:
    """pong メッセージを生成する.

    Args:
        client_ts_ms: 対応する ping の client_ts_ms をそのまま返す。

    Returns:
        pong メッセージ。
    """
    return {"type": "pong", "client_ts_ms": client_ts_ms, "server_ts_ms": now_ms()}


def make_error(code: str, message: str, fatal: bool) -> dict[str, Any]:
    """error メッセージを生成する.

    Args:
        code: PROTOCOL.md に定義されたエラーコード。
        message: エラー内容の説明。
        fatal: 送信後に接続を閉じるかどうか。

    Returns:
        error メッセージ。
    """
    return {"type": "error", "code": code, "message": message, "fatal": fatal}
