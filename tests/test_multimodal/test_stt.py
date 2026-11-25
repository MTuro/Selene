"""Speech-to-text fallback tests, extended for SELENE local-only operation."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from isaac.multimodal.voice.stt import STTManager


def test_stt_provider_fallback(monkeypatch):
    """Test that STTManager falls back to cloud providers if local fails."""
    audio_file = Path("test_audio.wav")
    monkeypatch.setattr("isaac.multimodal.voice.stt.WhisperModel", MagicMock())

    # We mock the LocalWhisperProvider in the STTManager's providers list
    with patch("pathlib.Path.exists", return_value=True):
        stt = STTManager(cloud_priority=["groq"])

        # Force the first provider (local) to fail if it exists,
        # or ensure the fallback logic works.
        # To properly test the chain, we can manually inject a failing provider.
        mock_fail = MagicMock()
        mock_fail.transcribe.side_effect = RuntimeError("Local failure")

        mock_success = MagicMock()
        mock_success.transcribe.return_value = "Hello from Groq"

        stt.providers = [mock_fail, mock_success]

        result = stt.transcribe(audio_file)

        assert result == "Hello from Groq"
        assert mock_fail.transcribe.called
        assert mock_success.transcribe.called


def test_stt_all_fail(monkeypatch):
    """Test that STTManager raises RuntimeError when all providers fail."""
    audio_file = Path("test_audio.wav")
    monkeypatch.setattr("isaac.multimodal.voice.stt.WhisperModel", MagicMock())

    with patch("pathlib.Path.exists", return_value=True):
        stt = STTManager(cloud_priority=["groq"])

        mock_fail1 = MagicMock()
        mock_fail1.transcribe.side_effect = RuntimeError("Fail 1")
        mock_fail2 = MagicMock()
        mock_fail2.transcribe.side_effect = RuntimeError("Fail 2")

        stt.providers = [mock_fail1, mock_fail2]

        with pytest.raises(RuntimeError, match="All STT providers failed"):
            stt.transcribe(audio_file)


def test_stt_file_not_found(monkeypatch):
    monkeypatch.setattr("isaac.multimodal.voice.stt.WhisperModel", MagicMock())
    """Test that STTManager raises FileNotFoundError for missing files."""
    stt = STTManager()
    with pytest.raises(FileNotFoundError):
        stt.transcribe(Path("non_existent_audio.wav"))


def test_get_stt_honors_voice_settings(monkeypatch):
    from isaac.config.settings import get_settings
    from isaac.multimodal.voice import stt as stt_module

    captured: dict[str, str] = {}

    class _FakeWhisperModel:
        def __init__(self, model: str, device: str, compute_type: str) -> None:
            captured.update(model=model, device=device, compute_type=compute_type)

    settings = get_settings()
    monkeypatch.setattr(settings, "voice_stt_model", "small")
    monkeypatch.setattr(settings, "voice_device", "cpu")
    monkeypatch.setattr(settings, "voice_stt_compute_type", "float32")
    monkeypatch.setattr(settings, "voice_stt_language", "pt")
    monkeypatch.setattr(stt_module, "WhisperModel", _FakeWhisperModel)
    monkeypatch.setattr(stt_module, "_stt_manager", None)

    manager = stt_module.get_stt()

    assert captured == {"model": "small", "device": "cpu", "compute_type": "float32"}
    assert manager.providers[0].language == "pt"  # type: ignore[attr-defined]


def test_get_stt_respects_voice_disabled(monkeypatch):
    from isaac.config.settings import get_settings
    from isaac.multimodal.voice import stt as stt_module

    monkeypatch.setattr(get_settings(), "voice_enabled", False)
    monkeypatch.setattr(stt_module, "_stt_manager", None)

    with pytest.raises(RuntimeError, match="disabled"):
        stt_module.get_stt()
    assert stt_module.is_stt_available() is False


def test_local_only_stt_never_reads_cloud_credentials(monkeypatch, tmp_path):
    from isaac.config.settings import get_settings
    from isaac.multimodal.voice import stt as stt_module

    monkeypatch.setenv("ISAAC_OPENAI_API_KEY", "test-cloud-key")
    credential = MagicMock(return_value="test-cloud-key")
    monkeypatch.setattr("isaac.security.credentials.get_credential", credential)
    monkeypatch.setattr(get_settings(), "voice_stt_allow_cloud", False)
    monkeypatch.setattr(stt_module, "_stt_manager", None)
    whisper = MagicMock()
    monkeypatch.setattr(stt_module, "WhisperModel", whisper)
    whisper.return_value.transcribe.side_effect = RuntimeError("local failure")
    audio_path = tmp_path / "prompt.wav"
    audio_path.touch()

    for manager in (stt_module.get_stt(), STTManager(cloud_priority=[])):
        assert len(manager.providers) == 1
        assert isinstance(manager.providers[0], stt_module.LocalWhisperProvider)
        with pytest.raises(RuntimeError, match="All STT providers failed"):
            manager.transcribe(audio_path)
    credential.assert_not_called()


@pytest.mark.parametrize("language", ["en", "pt"])
def test_selene_uses_hotwords_with_language_detection(monkeypatch, tmp_path, language):
    from isaac.config.settings import get_settings
    from isaac.multimodal.voice import stt as stt_module

    settings = get_settings()
    monkeypatch.setattr(settings, "identity_profile", "selene")
    monkeypatch.setattr(settings, "voice_stt_language", "")
    whisper = MagicMock()
    whisper.return_value.transcribe.return_value = (
        iter(
            [
                SimpleNamespace(
                    text=" Celine, how are you?" if language == "en" else " Celene, tudo bem?"
                )
            ]
        ),
        SimpleNamespace(language=language),
    )
    monkeypatch.setattr("isaac.multimodal.voice.stt.WhisperModel", whisper)
    audio_path = tmp_path / "prompt.wav"
    audio_path.touch()
    monkeypatch.setattr(stt_module, "_stt_manager", None)

    assert stt_module.get_stt().transcribe(audio_path).startswith(("Celine,", "Celene,"))
    whisper.return_value.transcribe.assert_called_once_with(
        str(audio_path), language=None, beam_size=5, hotwords="Selene"
    )
