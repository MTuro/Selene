"""TTS tests, extended for SELENE Edge voice and provider selection."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from isaac.multimodal.voice import tts as tts_module
from isaac.multimodal.voice.tts import (
    TTSManager,
    TTSProvider,
    _write_pcm_wav,
)


class FakeProvider(TTSProvider):
    audio_format = "mp3"
    supported_formats = frozenset({"mp3"})

    def __init__(self) -> None:
        self.paths: list[Path] = []

    async def synthesize(self, text: str, output_path: Path) -> Path:
        self.paths.append(output_path)
        output_path.write_bytes(b"ID3" + text.encode("utf-8"))
        return output_path


class FakeWavProvider(FakeProvider):
    audio_format = "wav"
    supported_formats = frozenset({"wav"})

    async def synthesize(self, text: str, output_path: Path) -> Path:
        self.paths.append(output_path)
        output_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        return output_path


@pytest.mark.asyncio
async def test_native_output_uses_provider_suffix() -> None:
    manager = TTSManager()
    provider = FakeProvider()
    manager._provider = provider
    manager._current_provider_name = "fake"

    result = await manager.synthesize("hello")

    assert result.suffix == ".mp3"
    assert result.read_bytes() == b"ID3hello"
    result.unlink()


@pytest.mark.asyncio
async def test_wav_target_is_converted_and_intermediate_removed(monkeypatch, tmp_path) -> None:
    module = types.ModuleType("soundfile")
    written: dict[str, object] = {}

    def _read(path: str, dtype: str, always_2d: bool):
        written["source"] = Path(path)
        return b"pcm-data", 22050

    def _write(path: str, data: object, samplerate: int, subtype: str) -> None:
        written["target"] = Path(path)
        written["data"] = data
        written["rate"] = samplerate
        Path(path).write_bytes(b"RIFFconverted")

    module.read = _read  # type: ignore[attr-defined]
    module.write = _write  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "soundfile", module)

    manager = TTSManager()
    provider = FakeProvider()
    manager._provider = provider
    manager._current_provider_name = "fake"
    target = tmp_path / "speech.wav"

    result = await manager.synthesize("hello", output_path=target)

    assert result == target
    assert target.read_bytes() == b"RIFFconverted"
    assert written["rate"] == 22050
    assert not provider.paths[0].exists()


@pytest.mark.asyncio
async def test_unsupported_target_format_fails_clearly() -> None:
    manager = TTSManager()
    provider = FakeProvider()
    manager._provider = provider
    manager._current_provider_name = "fake"

    with pytest.raises(RuntimeError, match=r"cannot produce \.ogg"):
        await manager.synthesize("hello", output_path=Path("speech.ogg"))
    assert provider.paths == []


@pytest.mark.asyncio
async def test_speak_plays_and_removes_owned_file(monkeypatch) -> None:
    played: list[str] = []

    def _play(path: str) -> None:
        played.append(path)
        assert Path(path).exists()

    monkeypatch.setattr("isaac.multimodal.voice.audio_io.play_wav", _play)

    manager = TTSManager()
    manager._provider = FakeWavProvider()
    manager._current_provider_name = "fake"

    await manager.speak("hello")

    assert len(played) == 1
    assert not Path(played[0]).exists()


def test_set_provider_rejects_unknown_backends() -> None:
    manager = TTSManager()

    with pytest.raises(ValueError, match="Supported: edge, elevenlabs, openai, piper"):
        manager.set_provider("gemini")

    manager.set_provider("piper")
    assert manager._current_provider_name == "piper"


def test_pcm_payload_is_wrapped_as_wav(tmp_path) -> None:
    target = tmp_path / "speech.wav"
    _write_pcm_wav(b"\x00\x01" * 10, target, sample_rate=44100)

    payload = target.read_bytes()
    assert payload.startswith(b"RIFF")
    assert payload[8:12] == b"WAVE"


@pytest.mark.parametrize("voice", ["pt-BR-ThalitaMultilingualNeural", "pt-BR-FranciscaNeural"])
async def test_edge_receives_the_selected_voice(monkeypatch, tmp_path, voice) -> None:
    from isaac.config.settings import get_settings

    monkeypatch.setattr(get_settings(), "voice_tts_edge_voice", voice)
    communicate = Mock(return_value=Mock(save=AsyncMock()))
    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=communicate))
    output = tmp_path / "speech.mp3"

    assert await TTSManager().synthesize("Hello", output) == output
    communicate.assert_called_once_with("Hello", voice)
    communicate.return_value.save.assert_awaited_once_with(str(output))


def test_thalita_is_the_default_edge_voice() -> None:
    from isaac.config.settings import Settings

    field = Settings.model_fields["voice_tts_edge_voice"]
    assert field.default == "pt-BR-ThalitaMultilingualNeural"


@pytest.mark.parametrize(
    ("name", "backend", "dependency"),
    [
        ("edge", tts_module.EdgeTTSProvider, "edge_tts"),
        ("piper", tts_module.PiperProvider, "piper"),
        ("openai", tts_module.OpenAIProvider, "openai"),
        ("elevenlabs", tts_module.ElevenLabsProvider, "httpx"),
    ],
)
def test_availability_uses_the_configured_backend(monkeypatch, name, backend, dependency) -> None:
    from isaac.config.settings import get_settings

    monkeypatch.setattr(get_settings(), "voice_tts_provider", name)
    monkeypatch.setattr(tts_module, "_manager", None)
    check_dependency = Mock()
    monkeypatch.setattr(tts_module, "import_module", check_dependency)

    assert tts_module.is_tts_available()
    check_dependency.assert_called_once_with(dependency)
    assert isinstance(tts_module.get_tts()._provider, backend)

    check_dependency.side_effect = ImportError(dependency)
    assert not tts_module.is_tts_available()


async def test_speak_cleans_up_when_edge_synthesis_fails(monkeypatch, tmp_path) -> None:
    output = tmp_path / "speech.wav"
    output.touch()
    manager = TTSManager()
    monkeypatch.setattr(tts_module, "_reserve_temp", Mock(return_value=output))
    monkeypatch.setattr(manager, "synthesize", AsyncMock(side_effect=OSError("offline")))

    with pytest.raises(OSError, match="offline"):
        await manager.speak("Olá")

    assert not output.exists()
