"""Terminal voice REPL with hands-free barge-in and push-to-talk modes."""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isaac.core.graph import build_graph
from isaac.core.state import make_initial_state
from isaac.identity.soul import assistant_name
from isaac.interfaces.realtime_voice import VoiceSession, listen_hands_free

if TYPE_CHECKING:
    from isaac.multimodal.voice.stt import STTManager

logger = logging.getLogger(__name__)


def _print(msg: str, *, color: str = "", end: str = "\n") -> None:
    if color and sys.stdout.isatty():
        sys.stdout.write(f"\033[{color}m{msg}\033[0m{end}")
    else:
        sys.stdout.write(msg + end)
    sys.stdout.flush()


def _meter(rms: float) -> None:
    width = 30
    level = int(min(1.0, rms * 4) * width)
    sys.stdout.write("\rListening [" + "#" * level + "-" * (width - level) + "]")
    sys.stdout.flush()


def _transcribe(stt: STTManager, audio: Any) -> str:
    from isaac.multimodal.voice.audio_io import save_wav

    with tempfile.TemporaryDirectory(prefix="isaac-utt-") as directory:
        path = str(Path(directory) / "utterance.wav")
        save_wav(audio, path)
        return stt.transcribe(path, local_only=True)


def _push_to_talk(session: VoiceSession) -> None:
    from isaac.multimodal.voice.audio_io import record_until_silence

    async def respond(audio: Any, latency: float | None) -> None:
        turn = session.start_utterance()
        session.submit(turn, audio, latency)
        await session.wait()

    with asyncio.Runner() as runner:
        try:
            while not session.exiting.is_set():
                try:
                    if input("[ENTER to record, /q to quit] ").strip().lower() == "/q":
                        break
                except EOFError:
                    break

                _print("Listening... ", color="36", end="")
                endpoint_latency = None

                def detected(latency: float) -> None:
                    nonlocal endpoint_latency
                    endpoint_latency = latency

                try:
                    audio = record_until_silence(on_chunk=_meter, on_endpoint=detected)
                except Exception as exc:
                    _print(f"\nMicrophone error: {exc}", color="31")
                    continue
                sys.stdout.write("\r" + " " * 60 + "\r")
                sys.stdout.flush()
                if len(audio) < 1000:
                    _print("(nothing heard)", color="33")
                    continue

                runner.run(respond(audio, endpoint_latency))
        finally:
            runner.run(session.close())


def run_voice_repl(hands_free: bool = False) -> int:
    """Launch terminal voice mode and return 2 if required backends are absent."""
    logging.basicConfig(level=logging.WARNING)
    for noisy in ("httpx", "apscheduler", "chromadb", "isaac"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("isaac.interfaces.realtime_voice").setLevel(logging.INFO)

    from isaac.config.settings import get_settings
    from isaac.multimodal.voice import is_stt_available, is_tts_available
    from isaac.multimodal.voice.audio_io import capture_utterances, is_audio_available, play_pcm
    from isaac.multimodal.voice.stt import get_stt
    from isaac.multimodal.voice.tts import get_tts

    settings = get_settings()
    if not settings.voice_enabled:
        _print("Voice REPL: voice subsystem is disabled (ISAAC_VOICE_ENABLED=false).", color="31")
        return 2
    if not is_audio_available():
        _print("Voice REPL: no audio device backend (install: sounddevice soundfile).", color="31")
        return 2

    _print("Loading STT model...", color="36")
    if not is_stt_available(local_only=True):
        _print("Voice REPL: local faster-whisper model could not be loaded.", color="31")
        return 2
    stt = get_stt()
    tts = get_tts() if is_tts_available() else None
    if tts is None:
        _print("Voice REPL: warning — no TTS backend; replies will be text-only.", color="33")

    try:
        from isaac.tools import register_all_tools

        register_all_tools()
    except Exception:
        logger.debug("Could not register tools", exc_info=True)

    state: dict[str, Any] = dict(make_initial_state())
    state.update(session_id=str(uuid.uuid4()), response_language="English only", voice_session=True)
    name = assistant_name()
    session = VoiceSession(
        build_graph(),
        state,
        lambda audio: _transcribe(stt, audio),
        tts.synthesize_pcm if tts is not None else None,
        play_pcm,
        _print,
        name=name,
        echo_tail_s=settings.voice_echo_tail_s,
        tts_timeout_s=settings.voice_tts_segment_timeout_s,
    )

    _print(f"\n[{name}] Voice REPL", color="1;36")
    try:
        if hands_free:
            _print("Hands-free mode. Speak when ready. Ctrl+C to exit.\n", color="36")
            asyncio.run(
                listen_hands_free(
                    session, lambda: capture_utterances(lambda: session.echo_gate, settings)
                )
            )
        else:
            _print("Push-to-talk mode. Press ENTER, speak, then stop talking.\n", color="36")
            _push_to_talk(session)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            from isaac.nodes.computer_use import shutdown_ui_executor

            shutdown_ui_executor()
        except Exception:
            logger.debug("Could not shut down UI executor", exc_info=True)
        _print("\nGoodbye.", color="36")

    return 0
