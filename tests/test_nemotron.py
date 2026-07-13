"""speech.nemotron モジュールのテスト.

NemotronStreamingRecognizerはGPU必須でこのMacでは実行できないため、ここでは
「NeMo未インストール環境でもimportしただけではサーバー本体が壊れない」ことと、
その状態でload_model()を呼ぶと明確なエラーになることだけを確認する。
"""

import pytest

from src.speech import nemotron


class TestImportSafety:
    """NeMo未インストール環境でのimport安全性のテスト."""

    def test_module_imports_without_nemo_installed(self):
        # このモジュール自体がimportできている時点で、nemoのimportが
        # トップレベルに漏れていないことの確認になる（このMacにはnemoがない）。
        assert nemotron.MODEL_NAME == "nvidia/nemotron-3.5-asr-streaming-0.6b"

    def test_load_model_without_nemo_raises_clear_runtime_error(self):
        with pytest.raises(RuntimeError, match="NeMo"):
            nemotron.NemotronStreamingRecognizer.load_model(560)

    def test_load_model_unsupported_chunk_ms_raises_value_error(self):
        with pytest.raises(ValueError):
            nemotron.NemotronStreamingRecognizer.load_model(999)

    def test_instantiate_without_load_model_raises_runtime_error(self):
        with pytest.raises(RuntimeError):
            nemotron.NemotronStreamingRecognizer(chunk_ms=560)
