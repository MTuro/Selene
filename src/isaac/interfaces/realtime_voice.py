"""Async turn ownership for the terminal voice interface."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterable, Awaitable, Callable, Iterator
from contextlib import aclosing, contextmanager
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage

from isaac.core.voice_turn import VoiceTurnInterrupted, current_voice_turn
from isaac.identity.address import normalize_selene_address, strip_selene_address
from isaac.identity.soul import enforce_speaking_identity, uses_selene_identity
from isaac.memory.context_manager import compress_messages

if TYPE_CHECKING:
    from isaac.multimodal.voice.audio_io import CaptureEvent

logger = logging.getLogger(__name__)
_SENTENCE_END = re.compile(r"""[.!?]+["'”’)]*(?=\s|$)|\n+""")
_ABBREVIATION = re.compile(r"(?:\b[A-Za-z]|\b(?:Mr|Ms|Mrs|Dr|Prof|vs|etc))\.$", re.I)
_SEGMENT_LIMIT = 280
_TEXT_AHEAD = 3
_AUDIO_AHEAD = 1
_STOP = re.compile(r"^stop(?:[.!?,]+\s*|$)", re.I)
_EXIT_WORDS = {"/q", "/quit", "/exit", "exit", "quit"}


class Phase(StrEnum):
    IDLE = auto()
    LISTENING = auto()
    TRANSCRIBING = auto()
    THINKING = auto()
    SPEAKING = auto()
    INTERRUPTED = auto()


def _metric(turn_id: str, metric: str, duration_s: float) -> None:
    event = {"event": metric, "turn_id": turn_id, "latency_ms": round(duration_s * 1000, 1)}
    logger.info("%s", json.dumps(event))


@dataclass
class Turn:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    spoken_command: str = ""
    approval: Any = None
    offer: Any = None
    phase: Phase = Phase.LISTENING
    interrupted_at: float | None = None
    is_barge_in: bool = False
    generation_task: asyncio.Task[Any] | None = None
    speech_task: asyncio.Task[None] | None = None
    submitted_at: float = 0
    stream_started: float = 0
    generation_started: float = 0
    first_token: bool = False
    first_audio: bool = False
    audio_failed: bool = False
    text_buffer: str = ""
    segments: asyncio.Queue[str | None] | None = None

    def check(self) -> None:
        if self.interrupted_at is not None:
            raise VoiceTurnInterrupted

    @contextmanager
    def generation(self) -> Iterator[None]:
        """Attach cancellation only during speech-producing LLM generation."""
        self.check()
        self.stream_started = time.monotonic()
        self.generation_task = asyncio.current_task()
        try:
            yield
        except asyncio.CancelledError:
            self.check()
            raise
        finally:
            self.generation_task = None

    async def token(self, text: str) -> None:
        self.check()
        if not self.first_token:
            self.first_token = True
            _metric(self.id, "first_llm_token", time.monotonic() - self.stream_started)
        await self._segment(text)

    async def finish(self, fallback: str) -> None:
        self.check()
        if self.segments is not None:
            await self._segment(fallback if not self.first_token else "", flush=True)
            await self.segments.put(None)

    async def _segment(self, text: str, *, flush: bool = False) -> None:
        if self.segments is None:
            return
        self.text_buffer += text
        while self.text_buffer:
            self.check()
            boundary = next(
                (
                    match.end()
                    for match in _SENTENCE_END.finditer(self.text_buffer)
                    if not _ABBREVIATION.search(self.text_buffer[: match.end()])
                ),
                None,
            )
            if boundary is None or boundary > _SEGMENT_LIMIT:
                if len(self.text_buffer) > _SEGMENT_LIMIT:
                    boundary = self.text_buffer.rfind(" ", 140, _SEGMENT_LIMIT)
                    if boundary == -1:
                        boundary = _SEGMENT_LIMIT
                elif flush:
                    boundary = len(self.text_buffer)
                else:
                    break
            sentence, self.text_buffer = self.text_buffer[:boundary], self.text_buffer[boundary:]
            if sentence := enforce_speaking_identity(sentence.strip()):
                await self.segments.put(sentence)


class VoiceSession:
    """Owns the active turn, one pending utterance, and one graph worker."""

    def __init__(
        self,
        graph: Any,
        state: dict[str, Any],
        transcribe: Callable[[Any], str],
        synthesize: Callable[[str], Awaitable[Any]] | None,
        play: Callable[[Any, Callable[[], None]], Awaitable[None]],
        report: Callable[[str], None],
        *,
        name: str = "SELENE",
        echo_tail_s: float = 0.15,
        tts_timeout_s: float = 15,
    ) -> None:
        self.graph = graph
        self.state = state
        self.transcribe = transcribe
        self.synthesize = synthesize
        self.play = play
        self.report = report
        self.name = name
        self.echo_tail_s = echo_tail_s
        self.tts_timeout_s = tts_timeout_s
        self.exiting = asyncio.Event()
        self.active: Turn | None = None
        self._worker: asyncio.Task[None] | None = None
        self._pending: tuple[Turn, Any] | None = None
        self._playing: Turn | None = None
        self._echo_until = 0.0

    @property
    def echo_gate(self) -> bool:
        return self._playing is not None or time.monotonic() < self._echo_until

    def start_utterance(self, detected_at: float | None = None) -> Turn:
        approval = self.active.offer if self.active and self.active.phase == Phase.IDLE else None
        self._invalidate(detected_at)
        self.active = Turn(approval=approval)
        return self.active

    def discard_capture(self, turn: Turn) -> None:
        if turn is self.active and turn.phase == Phase.LISTENING:
            self._invalidate()

    def submit(self, turn: Turn, audio: Any, endpoint_latency_s: float | None) -> None:
        if turn is not self.active or turn.phase != Phase.LISTENING:
            return
        if endpoint_latency_s is not None:
            _metric(turn.id, "speech_endpoint", endpoint_latency_s)
        turn.submitted_at = time.monotonic() - (endpoint_latency_s or 0)
        turn.phase = Phase.TRANSCRIBING
        self._pending = (turn, audio)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._process_pending(), name="voice-turn-worker")

    async def wait(self) -> None:
        if self._worker is not None:
            await asyncio.shield(self._worker)

    async def close(self) -> None:
        self._invalidate()
        await self.wait()

    def _invalidate(self, detected_at: float | None = None) -> None:
        self._pending = None
        turn = self.active
        if turn is not None:
            turn.offer = turn.approval = None
        if turn is None or turn.phase in {Phase.IDLE, Phase.INTERRUPTED}:
            return
        turn.interrupted_at = detected_at if detected_at is not None else time.monotonic()
        turn.phase = Phase.INTERRUPTED
        turn.is_barge_in = detected_at is not None
        turn.text_buffer = ""
        for task in (turn.generation_task, turn.speech_task):
            if task is not None:
                task.cancel()
        if turn.segments is not None:
            while not turn.segments.empty():
                turn.segments.get_nowait()
        if detected_at is not None and self._playing is not turn:
            _metric(turn.id, "barge_in_cancel", time.monotonic() - detected_at)

    async def _process_pending(self) -> None:
        while self._pending is not None:
            turn, audio = self._pending
            self._pending = None
            try:
                await self._respond(turn, audio)
            except VoiceTurnInterrupted:
                pass
            except Exception as exc:
                turn.offer = None
                if turn.interrupted_at is None:
                    self.report(f"[ERROR] {str(exc) or type(exc).__name__}")
            finally:
                if turn.phase != Phase.INTERRUPTED:
                    turn.phase = Phase.IDLE

    async def _respond(self, turn: Turn, audio: Any) -> None:
        started = time.monotonic()
        try:
            user_text = (await asyncio.to_thread(self.transcribe, audio)).strip()
        except Exception as exc:
            if turn.interrupted_at is None:
                self.report(f"STT error: {exc}")
            return
        finally:
            _metric(turn.id, "stt_transcription", time.monotonic() - started)
        turn.check()
        if not user_text:
            self.report("(no speech recognised)")
            return
        self.report(f">>> {user_text}")
        command = user_text
        if uses_selene_identity():
            normalized = normalize_selene_address(user_text)
            if normalized != user_text:
                self.report(f"Address recognized: {normalized}")
            user_text = normalized
            command = strip_selene_address(user_text)
        turn.spoken_command = command.strip()
        if stop := _STOP.match(command):
            turn.approval = None
            user_text = command[stop.end() :].strip()
            if not user_text:
                return
        if user_text.lower().rstrip(".!?") in _EXIT_WORDS:
            self.exiting.set()
            return

        from isaac.security.sanitizer import sanitize_input

        user_text = sanitize_input(user_text)
        messages = compress_messages(
            [*self.state.get("messages", []), HumanMessage(content=user_text)]
        )
        submitted = {**self.state, "messages": messages}
        turn.phase = Phase.THINKING
        turn.generation_started = time.monotonic()
        if self.synthesize is not None:
            turn.segments = asyncio.Queue(maxsize=_TEXT_AHEAD)
            turn.speech_task = asyncio.create_task(self._speech(turn), name="voice-speech")
        context = current_voice_turn.set(turn)
        try:
            result = submitted
            async with aclosing(self.graph.astream(submitted, stream_mode="values")) as events:
                async for snapshot in events:
                    result = snapshot
            turn.check()
            replies = result.get("messages", [])[len(messages) :]
            reply = next(
                (
                    text
                    for msg in reversed(replies)
                    if isinstance(msg, AIMessage) and (text := msg.text()).strip()
                ),
                "I'm here, but I have nothing to say about that.",
            )
            reply = enforce_speaking_identity(reply)
            if details := result.pop("project_result", None):
                self.report(json.dumps(details, ensure_ascii=True))
            self.state = result
            _metric(turn.id, "reply_text_ready", time.monotonic() - turn.generation_started)
            self.report(f"[{self.name}] {reply}\n")
            with turn.generation():
                await turn.finish(reply)
            if turn.speech_task is not None:
                try:
                    await turn.speech_task
                except asyncio.CancelledError:
                    turn.check()
                    raise
        finally:
            current_voice_turn.reset(context)
            if task := turn.speech_task:
                if not task.done() and turn.interrupted_at is None:
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                turn.speech_task = None
            turn.segments = None
            turn.text_buffer = ""

    async def _speech(self, turn: Turn) -> None:
        segments = turn.segments
        synthesize = self.synthesize
        assert segments is not None and synthesize is not None
        ready: asyncio.Queue[Any] = asyncio.Queue(maxsize=_AUDIO_AHEAD)

        async def produce() -> None:
            first_segment = True
            while turn.interrupted_at is None and (text := await segments.get()) is not None:
                if turn.audio_failed:
                    continue
                synthesis_start = time.monotonic()
                initial = first_segment
                if first_segment:
                    first_segment = False
                    _metric(
                        turn.id, "first_tts_segment", time.monotonic() - turn.generation_started
                    )
                try:
                    async with asyncio.timeout(self.tts_timeout_s):
                        audio = await synthesize(text)
                except Exception as exc:
                    turn.audio_failed = True
                    error = str(exc) or type(exc).__name__
                    self.report(f"TTS failed: {error} — reply available in text.")
                    continue
                if turn.interrupted_at is not None:
                    return
                if initial:
                    _metric(turn.id, "tts_first_audio_ready", time.monotonic() - synthesis_start)
                await ready.put(audio)
            if turn.interrupted_at is None:
                await ready.put(None)

        async def consume() -> None:
            while turn.interrupted_at is None and (audio := await ready.get()) is not None:
                if turn.interrupted_at is not None or turn.audio_failed:
                    continue
                self._playing = turn

                def on_start() -> None:
                    if turn.interrupted_at is None:
                        turn.phase = Phase.SPEAKING
                        if not turn.first_audio:
                            turn.first_audio = True
                            _metric(
                                turn.id, "response_latency", time.monotonic() - turn.submitted_at
                            )
                            _metric(
                                turn.id,
                                "first_spoken_audio",
                                time.monotonic() - turn.generation_started,
                            )

                try:
                    await self.play(audio, on_start)
                except Exception as exc:
                    turn.audio_failed = True
                    self.report(f"Playback failed: {str(exc) or type(exc).__name__}")
                finally:
                    self._playing = None
                    stopped = time.monotonic()
                    self._echo_until = stopped + self.echo_tail_s
                    if turn.is_barge_in and turn.interrupted_at is not None:
                        _metric(turn.id, "barge_in_cancel", stopped - turn.interrupted_at)
                    if turn.phase == Phase.SPEAKING:
                        turn.phase = Phase.THINKING

        async with asyncio.TaskGroup() as group:
            group.create_task(produce(), name="voice-synthesis")
            group.create_task(consume(), name="voice-playback")


async def listen_hands_free(
    session: VoiceSession,
    microphone: Callable[[], AsyncIterable[CaptureEvent]],
    *,
    retry_s: float = 0.5,
) -> None:
    """Keep opening the default mic until the session exits."""

    async def capture() -> None:
        while True:
            turn = None
            try:
                async for event in microphone():
                    if event.audio is None:
                        turn = session.start_utterance(event.at)
                        if event.onset_latency_s is not None:
                            _metric(turn.id, "vad_onset_window", event.onset_latency_s)
                    elif turn is not None:
                        session.submit(turn, event.audio, event.endpoint_latency_s)
                        turn = None
                raise OSError("Microphone stream ended")
            except Exception as exc:
                if turn is not None:
                    session.discard_capture(turn)
                session.report(f"Microphone error: {exc} — retrying.")
                await asyncio.sleep(retry_s)

    task = asyncio.create_task(capture(), name="voice-microphone")
    try:
        await session.exiting.wait()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await session.close()
