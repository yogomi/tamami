"""server.protocol モジュールのテスト."""

import pytest

from src.server import protocol


class TestParseClientMessage:
    """parse_client_message のテスト."""

    def test_valid_message(self):
        message = protocol.parse_client_message('{"type": "ping", "client_ts_ms": 123}')
        assert message["type"] == "ping"
        assert message["client_ts_ms"] == 123

    def test_invalid_json_raises(self):
        with pytest.raises(protocol.ProtocolError) as exc_info:
            protocol.parse_client_message("{not json")
        assert exc_info.value.code == "invalid_config"
        assert exc_info.value.fatal is True

    def test_non_object_raises(self):
        with pytest.raises(protocol.ProtocolError) as exc_info:
            protocol.parse_client_message('["a", "b"]')
        assert exc_info.value.code == "invalid_config"

    def test_missing_type_raises(self):
        with pytest.raises(protocol.ProtocolError) as exc_info:
            protocol.parse_client_message('{"client_ts_ms": 123}')
        assert exc_info.value.code == "invalid_config"


class TestValidateSessionStart:
    """validate_session_start のテスト."""

    def test_valid_session_start(self):
        message = {
            "type": "session_start",
            "protocol_version": protocol.PROTOCOL_VERSION,
            "client_ts_ms": 1751400000000,
        }
        protocol.validate_session_start(message)

    def test_unsupported_version_raises(self):
        message = {"type": "session_start", "protocol_version": 999, "client_ts_ms": 0}
        with pytest.raises(protocol.ProtocolError) as exc_info:
            protocol.validate_session_start(message)
        assert exc_info.value.code == "unsupported_version"
        assert exc_info.value.fatal is True

    def test_missing_version_raises(self):
        message = {"type": "session_start", "client_ts_ms": 0}
        with pytest.raises(protocol.ProtocolError) as exc_info:
            protocol.validate_session_start(message)
        assert exc_info.value.code == "unsupported_version"

    def test_missing_client_ts_raises(self):
        message = {
            "type": "session_start",
            "protocol_version": protocol.PROTOCOL_VERSION,
        }
        with pytest.raises(protocol.ProtocolError) as exc_info:
            protocol.validate_session_start(message)
        assert exc_info.value.code == "invalid_config"


class TestMessageBuilders:
    """下りメッセージ生成のテスト."""

    def test_session_ready_fields(self):
        message = protocol.make_session_ready("abc123")
        assert message["type"] == "session_ready"
        assert message["protocol_version"] == protocol.PROTOCOL_VERSION
        assert message["session_id"] == "abc123"
        assert isinstance(message["server_ts_ms"], int)

    def test_asr_fields(self):
        message = protocol.make_asr(12, "こんにちは", "ja", False, 34.2, 36.8)
        assert message["type"] == "asr"
        assert message["segment_id"] == 12
        assert message["text"] == "こんにちは"
        assert message["lang"] == "ja"
        assert message["is_final"] is False
        assert message["ts_audio_start"] == 34.2
        assert message["ts_audio_end"] == 36.8
        assert isinstance(message["server_ts_ms"], int)

    def test_translation_fields(self):
        message = protocol.make_translation(12, "привіт", "ja", "uk", True, 34.2, 36.8)
        assert message["type"] == "translation"
        assert message["source_lang"] == "ja"
        assert message["target_lang"] == "uk"
        assert message["is_final"] is True

    def test_pong_echoes_client_ts(self):
        message = protocol.make_pong(1751400000000)
        assert message["type"] == "pong"
        assert message["client_ts_ms"] == 1751400000000
        assert isinstance(message["server_ts_ms"], int)

    def test_error_fields(self):
        message = protocol.make_error("webrtc_failure", "boom", True)
        assert message["type"] == "error"
        assert message["code"] == "webrtc_failure"
        assert message["fatal"] is True
