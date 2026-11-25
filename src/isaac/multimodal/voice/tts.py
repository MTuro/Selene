"""Text-to-Speech Manager with Provider Matrix.

Supports Edge (default), OpenAI, ElevenLabs, and Piper.
Modified for SELENE to configure voice and select the playback provider.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
import os
import tempfile
import wave
from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

_OWNED_TEMP_FILES: set[Path] = set()


def _cleanup_temp_files() -> None:
    for path in list(_OWNED_TEMP_FILES):
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        _OWNED_TEMP_FILES.discard(path)


atexit.register(_cleanup_temp_files)


def _reserve_temp(suffix: str) -> Path:
    descriptor, filename = tempfile.mkstemp(prefix="isaac-tts-", suffix=suffix)
    os.close(descriptor)
    path = Path(filename)
    _OWNED_TEMP_FILES.add(path)
    return path


def _discard_temp(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove temporary TTS file %s", path)
    _OWNED_TEMP_FILES.discard(path)


def _write_pcm_wav(pcm: bytes, output_path: Path, sample_rate: int) -> None:
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def _convert_to_wav(source: Path, target: Path) -> Path:
    try:
        import soundfile as sf
    except ImportError as err:
        raise RuntimeError(
            "Converting TTS audio to WAV requires soundfile: pip install soundfile"
        ) from err
    try:
        data, sample_rate = sf.read(str(source), dtype="int16", always_2d=False)
        sf.write(str(target), data, sample_rate, subtype="PCM_16")
    except Exception as exc:
        raise RuntimeError(f"Could not convert TTS audio to WAV: {exc}") from exc
    return target


class TTSProvider(ABC):
    """Base adapter for TTS providers."""

    audio_format: ClassVar[str] = "mp3"
    supported_formats: ClassVar[frozenset[str]] = frozenset({"mp3"})
    dependency: ClassVar[str]

    @abstractmethod
    async def synthesize(self, text: str, output_path: Path) -> Path:
        pass


class EdgeTTSProvider(TTSProvider):
    """Free, streaming TTS using edge-tts."""

    dependency = "edge_tts"

    def __init__(self, voice: str | None = None) -> None:
        from isaac.config.settings import get_settings

        self.voice = voice if voice is not None else get_settings().voice_tts_edge_voice

    async def synthesize(self, text: str, output_path: Path) -> Path:
        try:
            import edge_tts
        except ImportError as err:
            raise RuntimeError("edge-tts not installed. Run `pip install edge-tts`.") from err
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(str(output_path))
        return output_path


class OpenAIProvider(TTSProvider):
    """OpenAI TTS API."""

    dependency = "openai"
    supported_formats = frozenset({"mp3", "opus", "aac", "flac", "wav", "pcm"})

    async def synthesize(self, text: str, output_path: Path) -> Path:
        try:
            from openai import AsyncOpenAI
        except ImportError as err:
            raise RuntimeError("openai not installed.") from err

        from isaac.config.settings import settings
        from isaac.security.credentials import get_credential

        api_key = get_credential("openai") or settings.openai_api_key.strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured for the openai TTS provider.")

        request: dict[str, Any] = {"model": "tts-1", "voice": "alloy", "input": text}
        audio_format = output_path.suffix.lstrip(".").lower()
        if audio_format in self.supported_formats:
            request["response_format"] = audio_format

        async with AsyncOpenAI(api_key=api_key) as client:
            response = await client.audio.speech.create(**request)
            output_path.write_bytes(await response.aread())
        return output_path


class ElevenLabsProvider(TTSProvider):
    """ElevenLabs TTS API."""

    dependency = "httpx"
    supported_formats = frozenset({"mp3", "wav"})

    async def synthesize(self, text: str, output_path: Path) -> Path:
        try:
            import httpx
        except ImportError as err:
            raise RuntimeError("httpx not installed.") from err

        from isaac.config.settings import settings

        if not settings.elevenlabs_api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY is not configured for the elevenlabs TTS provider."
            )

        audio_format = output_path.suffix.lstrip(".").lower() or "mp3"
        params: dict[str, str] = {}
        if audio_format == "wav":
            params["output_format"] = "pcm_44100"
        elif audio_format != "mp3":
            raise RuntimeError(f"ElevenLabs TTS does not support .{audio_format} output.")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{settings.elevenlabs_voice_id}",
                params=params,
                headers={"xi-api-key": settings.elevenlabs_api_key},
                json={"text": text},
            )
            response.raise_for_status()
            payload = response.content

        if audio_format == "wav":
            _write_pcm_wav(payload, output_path, sample_rate=44100)
        else:
            output_path.write_bytes(payload)
        return output_path


def _piper_voice_path(configured_voice: str, isaac_home: Path) -> Path:
    """Resolve a Piper voice name or path to an ``.onnx`` model file."""
    configured = Path(configured_voice).expanduser()
    candidates = [configured]
    if not configured.is_absolute():
        base = Path(os.environ.get("PIPER_VOICE_DIR", "") or isaac_home / "voices")
        candidates.append(base / configured)
        if configured.suffix != ".onnx":
            candidates.append(base / f"{configured}.onnx")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        f"Piper voice model not found: {configured_voice}. Place the .onnx file under "
        f"{Path.home() / '.isaac' / 'voices'} or set PIPER_VOICE_DIR."
    )


class PiperProvider(TTSProvider):
    """Local Piper TTS."""

    dependency = "piper"
    audio_format = "wav"
    supported_formats = frozenset({"wav"})

    async def synthesize(self, text: str, output_path: Path) -> Path:
        try:
            from piper import PiperVoice
        except ImportError as err:
            raise RuntimeError("piper-tts not installed.") from err

        from isaac.config.settings import settings

        model_path = str(_piper_voice_path(settings.voice_tts_voice, settings.isaac_home))

        def render() -> None:
            self._write_wav(PiperVoice.load(model_path), text, output_path)

        render_task = asyncio.create_task(asyncio.to_thread(render))
        try:
            await asyncio.shield(render_task)
        except asyncio.CancelledError:
            # The native renderer cannot be interrupted. Join it before its file is removed.
            while not render_task.done():
                try:
                    await asyncio.shield(render_task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not render_task.cancelled():
                render_task.exception()
            raise
        return output_path

    @staticmethod
    def _write_wav(voice: Any, text: str, output_path: Path) -> None:
        with wave.open(str(output_path), "wb") as wav:
            sample_rate = int(getattr(voice.config, "sample_rate", 22050))
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            voice.synthesize_wav(text, wav)


class TTSManager:
    """Coordinates TTS synthesis across multiple providers."""

    PROVIDERS: dict[str, type[TTSProvider]] = {
        "edge": EdgeTTSProvider,
        "openai": OpenAIProvider,
        "elevenlabs": ElevenLabsProvider,
        "piper": PiperProvider,
    }

    def __init__(self) -> None:
        from isaac.config.settings import get_settings

        self.set_provider(get_settings().voice_tts_provider)

    def _load(self) -> None:
        """Compatibility hook; providers initialize on first synthesis."""
        return None

    def set_provider(self, name: str) -> None:
        """Select the backend used for synthesis."""
        if name not in self.PROVIDERS:
            supported = ", ".join(sorted(self.PROVIDERS))
            raise ValueError(f"Unsupported TTS provider: {name}. Supported: {supported}.")
        self._provider = self.PROVIDERS[name]()
        self._current_provider_name = name
        logger.info("TTS provider switched to %s", name)

    def is_available(self) -> bool:
        """Check whether the selected backend can be imported."""
        try:
            import_module(self._provider.dependency)
        except (ImportError, OSError):
            return False
        return True

    async def synthesize(self, text: str, output_path: Path | None = None) -> Path:
        provider = self._provider
        requested = Path(output_path) if output_path is not None else None
        audio_format = (
            requested.suffix.lstrip(".").lower()
            if requested is not None and requested.suffix
            else provider.audio_format
        )

        if requested is not None and audio_format in provider.supported_formats:
            return await provider.synthesize(text, requested)

        target = requested if requested is not None else _reserve_temp(f".{audio_format}")
        if audio_format == provider.audio_format:
            return await provider.synthesize(text, target)
        if audio_format != "wav":
            raise RuntimeError(
                f"The {self._current_provider_name} TTS provider cannot produce "
                f".{audio_format} audio."
            )

        intermediate = _reserve_temp(f".{provider.audio_format}")
        try:
            produced = await provider.synthesize(text, intermediate)
            return _convert_to_wav(produced, target)
        finally:
            _discard_temp(intermediate)

    async def synthesize_pcm(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize one segment and decode it for cancellable PortAudio playback."""
        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="isaac-voice-tts-") as directory:
            path = Path(directory) / f"segment.{self._provider.audio_format}"
            produced = await self.synthesize(text, output_path=path)
            return sf.read(produced, dtype="float32", always_2d=True)

    async def speak(self, text: str) -> None:
        """Synthesize speech and play it through the default output device."""
        from isaac.multimodal.voice.audio_io import play_wav

        path = _reserve_temp(".wav")
        try:
            produced = await self.synthesize(text, output_path=path)
            await asyncio.to_thread(play_wav, str(produced))
        finally:
            _discard_temp(path)


_manager: TTSManager | None = None


def get_tts_manager() -> TTSManager:
    global _manager
    if _manager is None:
        _manager = TTSManager()
    return _manager


class TextToSpeech:
    def __init__(self) -> None:
        self._manager = TTSManager()

    def synthesize(self, text: str, out_path: Path | None = None) -> Path:
        """Synchronous compatibility wrapper for legacy REPL integrations."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._manager.synthesize(text, output_path=out_path))
        raise RuntimeError("Use TTSManager.synthesize from async code.")


def get_tts() -> TTSManager:
    return get_tts_manager()


def is_tts_available() -> bool:
    """True when voice is enabled and the selected TTS dependency can load."""
    from isaac.config.settings import get_settings

    return get_settings().voice_enabled and get_tts().is_available()
