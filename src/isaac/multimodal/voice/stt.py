"""Speech-to-Text (STT) management.

Modified for SELENE to make conversational STT local by default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from isaac.config.settings import Settings

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

logger = logging.getLogger(__name__)


class STTProvider(Protocol):
    """Interface for an STT provider."""

    def transcribe(self, audio_path: Path) -> str: ...


class LocalWhisperProvider:
    """Local STT using faster-whisper."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        language: str = "",
        hotwords: str = "",
    ):
        if WhisperModel is None:
            raise ImportError(
                "faster-whisper not installed. Install with `pip install faster-whisper`."
            )

        logger.info("Initializing local Whisper model (%s) on %s...", model_size, device)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language or None
        self.hotwords = hotwords

    def transcribe(self, audio_path: Path) -> str:
        options = {"hotwords": self.hotwords} if self.hotwords else {}
        segments, _ = self.model.transcribe(
            str(audio_path), language=self.language, beam_size=5, **options
        )
        return " ".join(segment.text for segment in segments).strip()


class OpenAISTTProvider:
    """Cloud STT through the OpenAI audio-transcriptions endpoint."""

    def __init__(self, api_key: str, model: str = "whisper-1") -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")
        self._api_key = api_key
        self._model = model

    def transcribe(self, audio_path: Path) -> str:
        from openai import OpenAI

        with OpenAI(api_key=self._api_key) as client, audio_path.open("rb") as handle:
            result = client.audio.transcriptions.create(model=self._model, file=handle)
        return str(getattr(result, "text", "")).strip()


class STTManager:
    """Manages a priority chain of STT providers."""

    def __init__(
        self,
        local_model_size: str = "base",
        cloud_priority: list[str] | None = None,
        device: str = "auto",
        compute_type: str = "int8",
        language: str = "",
        hotwords: str = "",
    ):
        self.cloud_priority = ["openai"] if cloud_priority is None else cloud_priority
        self.providers: list[STTProvider] = []

        try:
            self.providers.append(
                LocalWhisperProvider(
                    model_size=local_model_size,
                    device=device,
                    compute_type=compute_type,
                    language=language,
                    hotwords=hotwords,
                )
            )
        except Exception as exc:
            logger.warning("Could not initialize local STT: %s", exc)

        if "openai" in {name.lower() for name in self.cloud_priority}:
            from isaac.security.credentials import get_credential

            api_key = get_credential("openai")
            if api_key:
                self.providers.append(OpenAISTTProvider(api_key))

    def _load(self) -> None:
        """Compatibility hook; providers are prepared during construction."""
        return None

    def transcribe(self, audio_path: str | Path, *, local_only: bool = False) -> str:
        """Transcribe audio using the first available provider in the chain."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        for provider in self.providers:
            if local_only and not isinstance(provider, LocalWhisperProvider):
                continue
            try:
                text = provider.transcribe(audio_path)
            except Exception as exc:
                logger.error("STT provider %s failed: %s", provider.__class__.__name__, exc)
                continue
            if local_only or text.strip():
                return text
            logger.warning("STT provider %s returned no text.", provider.__class__.__name__)

        raise RuntimeError("All STT providers failed or no providers configured.")


# Aliases to satisfy existing __init__.py imports
SpeechToText = STTManager


_stt_manager: STTManager | None = None
_stt_settings: Settings | None = None


def get_stt() -> STTManager:
    """Singleton-like getter for STT manager, honoring the voice settings."""
    from isaac.config.settings import get_settings

    global _stt_manager, _stt_settings
    settings = get_settings()
    if not settings.voice_enabled:
        raise RuntimeError("Voice subsystem is disabled (ISAAC_VOICE_ENABLED=false).")
    if _stt_manager is None or _stt_settings is not settings:
        _stt_manager = STTManager(
            local_model_size=settings.voice_stt_model,
            cloud_priority=None if settings.voice_stt_allow_cloud else [],
            device=settings.voice_device,
            compute_type=settings.voice_stt_compute_type,
            language=settings.voice_stt_language,
            hotwords="Selene" if settings.identity_profile == "selene" else "",
        )
        _stt_settings = settings
    return _stt_manager


def is_stt_available(*, local_only: bool = False) -> bool:
    """Check whether at least one real STT backend is configured."""
    from isaac.config.settings import get_settings

    if not get_settings().voice_enabled:
        return False
    try:
        return any(
            not local_only or isinstance(provider, LocalWhisperProvider)
            for provider in get_stt().providers
        )
    except Exception:
        return False
