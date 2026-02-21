"""
Tests for ptt_gui.core.load_model().

The actual Whisper model download is mocked so tests run without network
or large model files.
"""

from unittest.mock import patch, MagicMock

import pytest

from ptt_gui.core import load_model, DEFAULT_WHISPER_MODEL, DEFAULT_LANGUAGE


class TestLoadModel:
    @patch("ptt_gui.core.WhisperModel")
    def test_returns_model(self, mock_cls):
        """load_model() should return a WhisperModel instance."""
        fake_model = MagicMock()
        mock_cls.return_value = fake_model

        result = load_model()

        assert result is fake_model

    @patch("ptt_gui.core.WhisperModel")
    def test_uses_cpu_device(self, mock_cls):
        """load_model() should request CPU device."""
        mock_cls.return_value = MagicMock()

        load_model()

        _, kwargs = mock_cls.call_args
        assert kwargs.get("device") == "cpu"

    @patch("ptt_gui.core.WhisperModel")
    def test_uses_int8_compute_type(self, mock_cls):
        """load_model() should use int8 quantisation for fastest CPU inference."""
        mock_cls.return_value = MagicMock()

        load_model()

        _, kwargs = mock_cls.call_args
        assert kwargs.get("compute_type") == "int8"

    @patch("ptt_gui.core.WhisperModel")
    def test_default_model_size(self, mock_cls):
        """load_model() should use DEFAULT_WHISPER_MODEL ('base') by default."""
        mock_cls.return_value = MagicMock()

        load_model()

        args, _ = mock_cls.call_args
        assert args[0] == DEFAULT_WHISPER_MODEL

    @patch("ptt_gui.core.WhisperModel")
    def test_custom_model_size_passed_through(self, mock_cls):
        """load_model('small') should create WhisperModel('small', ...)."""
        mock_cls.return_value = MagicMock()

        load_model("small")

        args, _ = mock_cls.call_args
        assert args[0] == "small"

    @patch("ptt_gui.core.WhisperModel")
    def test_language_attribute_set_on_model(self, mock_cls):
        """load_model() should attach a .language attribute to the returned model."""
        fake_model = MagicMock(spec=[])  # empty spec so we can set arbitrary attrs
        mock_cls.return_value = fake_model

        result = load_model()

        assert hasattr(result, "language")
        assert result.language == DEFAULT_LANGUAGE

    @patch("ptt_gui.core.WhisperModel")
    def test_custom_language(self, mock_cls):
        """load_model(language='en') should set .language = 'en'."""
        fake_model = MagicMock(spec=[])
        mock_cls.return_value = fake_model

        result = load_model(language="en")

        assert result.language == "en"

    def test_default_whisper_model_is_defined(self):
        """DEFAULT_WHISPER_MODEL should be a non-empty string."""
        assert DEFAULT_WHISPER_MODEL
        assert isinstance(DEFAULT_WHISPER_MODEL, str)

    def test_default_language_is_polish(self):
        """DEFAULT_LANGUAGE should be 'pl'."""
        assert DEFAULT_LANGUAGE == "pl"
