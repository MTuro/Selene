"""Turn-local bridge between terminal voice and the cognitive graph."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from contextlib import AbstractContextManager


class VoiceTurnInterrupted(Exception):
    """A voice turn was superseded before its graph could complete."""


class VoiceStream(Protocol):
    def check(self) -> None: ...
    def generation(self) -> AbstractContextManager[None]: ...
    async def token(self, text: str) -> None: ...


current_voice_turn: ContextVar[VoiceStream | None] = ContextVar("current_voice_turn", default=None)


def check_voice_turn() -> None:
    if turn := current_voice_turn.get():
        turn.check()
