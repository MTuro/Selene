"""DirectResponse Node — fast-path for simple conversational queries.

When Perception detects that the user's request is a simple question,
greeting, or conversational message that does NOT require code execution,
this node generates a direct LLM response and routes to END — bypassing
Explorer, Planner, Synthesis, Sandbox, and Reflection entirely.

This is the single most impactful optimisation: it reduces a 6+ LLM call
pipeline to a **single** LLM call with streaming output.

Modified for SELENE to use the voice session's requested reply language.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from isaac.core.state import IsaacState
from isaac.core.voice_turn import check_voice_turn, current_voice_turn
from isaac.identity.address import selene_identity_reply
from isaac.identity.soul import assistant_name, enforce_speaking_identity, uses_selene_identity

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from langchain_core.messages import BaseMessageChunk

logger = logging.getLogger(__name__)


def _build_direct_prompt(
    user_text: str,
    hypothesis: str,
    history: list[Any] | None = None,
    response_language: str = "",
    *,
    spoken: bool = False,
) -> list[Any]:
    """Build a minimal prompt for direct conversational response."""
    from isaac.identity.soul import soul_system_prompt

    soul = ""
    with contextlib.suppress(Exception):
        soul = soul_system_prompt()

    system_content = (
        f"{soul}\n\n"
        "You are in DIRECT RESPONSE mode.  The user's message is conversational "
        "and does NOT require code execution, planning, or tool use.  Reply "
        "naturally, concisely, and helpfully.  Be warm but efficient.  "
        "If the user greets you, greet them back and ask how you can help.  "
        "If they ask a knowledge question, answer it directly.  "
        "Keep responses under 200 words unless the topic demands more."
    )
    if response_language:
        system_content += f" Reply in {response_language}."
    if spoken:
        system_content += (
            " Start with a complete sentence of 12 words or fewer that answers the question."
            " Elaborate in short sentences using plain speech."
        )
    messages: list[Any] = [SystemMessage(content=system_content)]

    if hypothesis:
        messages.append(SystemMessage(content=f"Context from perception: {hypothesis[:300]}"))

    messages.extend((history or [])[-10:])
    messages.append(HumanMessage(content=user_text))
    return messages


def _latest_question(state: IsaacState) -> tuple[str, list[Any]]:
    messages = state.get("messages", [])
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(message := messages[index], HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content), messages[:index]
    return "", list(messages)


async def async_direct_response_node(state: IsaacState) -> dict[str, Any]:
    """Expose only public answer text from the native async model stream."""
    turn = current_voice_turn.get()
    if turn is None:
        return await asyncio.to_thread(direct_response_node, state)

    from isaac.config.settings import get_settings
    from isaac.llm.provider import get_direct_response_llm

    check_voice_turn()
    user_text, history = _latest_question(state)
    answer = selene_identity_reply(user_text)
    if not answer:
        # The existing provider factory performs a synchronous Ollama preflight.
        llm = await asyncio.to_thread(get_direct_response_llm)
        prompt = _build_direct_prompt(
            user_text,
            state.get("hypothesis", ""),
            history,
            state.get("response_language", ""),
            spoken=True,
        )
        parts: list[str] = []
        timeout = get_settings().voice_llm_token_timeout_s
        # LangChain annotates astream as AsyncIterator, but its implementation is a generator.
        response = cast("AsyncGenerator[BaseMessageChunk, None]", llm.astream(prompt))
        with turn.generation():
            async with contextlib.aclosing(response) as stream:
                while True:
                    try:
                        async with asyncio.timeout(timeout):
                            chunk = await anext(stream)
                    except StopAsyncIteration:
                        break
                    except TimeoutError as exc:
                        raise TimeoutError("Timed out waiting for a model response.") from exc
                    if token := chunk.text():
                        parts.append(token)
                        await turn.token(token)
        answer = "".join(parts)
    return {
        "messages": [AIMessage(content=enforce_speaking_identity(answer))],
        "current_phase": "direct_response",
    }


def direct_response_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: DirectResponse — single-call fast-path.

    Streams the LLM response token-by-token.  When the Rich terminal UI
    is active it streams through that; otherwise falls back to raw stdout.
    """
    from isaac.llm.provider import get_direct_response_llm

    user_text, history = _latest_question(state)

    # Resolve the active UI (if running inside the Rich REPL)
    ui = None
    try:
        from isaac.interfaces.repl import get_active_ui

        ui = get_active_ui()
    except ImportError:
        pass

    selene = uses_selene_identity()
    show = not state.get("voice_session", False)
    name = assistant_name()
    if answer := selene_identity_reply(user_text):
        if show:
            if ui is not None:
                ui.start_stream()
                ui.stream_token(answer)
            else:
                print(f"\n[{name}] {answer}\n")
        return {"messages": [AIMessage(content=answer)], "current_phase": "direct_response"}

    llm = get_direct_response_llm()
    prompt = _build_direct_prompt(
        user_text, state.get("hypothesis", ""), history, state.get("response_language", "")
    )

    # Try streaming for instant feedback
    full_response = ""
    try:
        if show and ui is not None:
            ui.start_stream()
        elif show:
            sys.stdout.write(f"\n[{name}] ")
            sys.stdout.flush()

        for chunk in llm.stream(prompt):
            raw_token = chunk.content if hasattr(chunk, "content") else str(chunk)
            token = raw_token if isinstance(raw_token, str) else str(raw_token)
            if token:
                if show and not selene and ui is not None:
                    ui.stream_token(token)
                elif show and not selene:
                    sys.stdout.write(token)
                    sys.stdout.flush()
                full_response += token

        full_response = enforce_speaking_identity(full_response)
        if show:
            if ui is None:
                if selene:
                    sys.stdout.write(full_response)
                sys.stdout.write("\n\n")
                sys.stdout.flush()
            elif selene:
                ui.stream_token(full_response)

    except (AttributeError, TypeError):
        # Fallback: non-streaming
        response = llm.invoke(prompt)
        full_response = (
            response.content if isinstance(response.content, str) else str(response.content)
        )
        full_response = enforce_speaking_identity(full_response)
        if show:
            if ui is not None:
                ui.start_stream()
                ui.stream_token(full_response)
            else:
                print(f"\n[{name}] {full_response}\n")

    logger.info("DirectResponse: %d chars (fast-path).", len(full_response))

    return {
        "messages": [AIMessage(content=full_response)],
        "current_phase": "direct_response",
    }
