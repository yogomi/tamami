"""speech.nemotron モジュールのテスト.

NemotronStreamingRecognizerはGPU前提でこのMacでは実行できないため、ここでは
「transformers未インストール環境でもimportしただけではサーバー本体が壊れない」
ことと、その状態でload_model()を呼ぶと明確なエラーになることだけを確認する。
GPU上での実動作はDGX Spark側で検証する（DGX_SETUP.md参照）。
"""

import importlib.util

import pytest

from src.speech import nemotron

_TRANSFORMERS_INSTALLED = importlib.util.find_spec("transformers") is not None


class TestImportSafety:
    """transformers未インストール環境でのimport安全性のテスト."""

    def test_module_imports_without_transformers_installed(self):
        # このモジュール自体がimportできている時点で、transformers/torchの
        # importがトップレベルに漏れていないことの確認になる。
        assert nemotron.MODEL_NAME == "nvidia/nemotron-3.5-asr-streaming-0.6b"

    @pytest.mark.skipif(
        _TRANSFORMERS_INSTALLED,
        reason="transformersがインストール済みの環境ではImportError経路を通らない",
    )
    def test_load_model_without_transformers_raises_clear_runtime_error(self):
        with pytest.raises(RuntimeError, match="transformers"):
            nemotron.NemotronStreamingRecognizer.load_model(560)

    def test_load_model_unsupported_chunk_ms_raises_value_error(self):
        with pytest.raises(ValueError):
            nemotron.NemotronStreamingRecognizer.load_model(999)

    def test_load_model_chunk_ms_160_raises_value_error(self):
        # 旧NeMoドラフトは160msに対応していたが、Transformers経路の
        # lookahead段階（80/320/560/1120）に160はない。
        with pytest.raises(ValueError):
            nemotron.NemotronStreamingRecognizer.load_model(160)

    def test_instantiate_without_load_model_raises_runtime_error(self):
        with pytest.raises(RuntimeError):
            nemotron.NemotronStreamingRecognizer(chunk_ms=560)


class TestDetectLangFromScript:
    """文字種による言語推定（言語タグ欠落時のフォールバック）のテスト."""

    def test_ukrainian_text_returns_uk(self):
        assert nemotron.detect_lang_from_script("Потім Лака Синг взяв на себе") == "uk"

    def test_japanese_text_returns_ja(self):
        assert nemotron.detect_lang_from_script("多くの場合、大学に進学しやすくなります。") == "ja"

    def test_katakana_only_returns_ja(self):
        assert nemotron.detect_lang_from_script("ギャップイヤーコース") == "ja"

    def test_latin_or_empty_returns_none(self):
        assert nemotron.detect_lang_from_script("hello world 123") is None
        assert nemotron.detect_lang_from_script("") is None
