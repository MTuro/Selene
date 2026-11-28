"""Real-time voice behavior with externally controlled graph and audio I/O."""

import asyncio

import pytest
from langchain_core.messages import AIMessage

from isaac.core.voice_turn import current_voice_turn
from isaac.interfaces.realtime_voice import VoiceSession


@pytest.mark.asyncio
async def test_speaks_in_order_before_graph_finishes():
    heard_first = asyncio.Event()
    played = []

    class Graph:
        async def astream(self, state, *, stream_mode):
            assert stream_mode == "values"
            turn = current_voice_turn.get()
            await turn.token("Containers share the host kernel. ")
            await asyncio.wait_for(heard_first.wait(), 1)
            await turn.token("They isolate each application.")
            yield {
                "messages": [
                    *state["messages"],
                    AIMessage(
                        content="Containers share the host kernel. They isolate each application."
                    ),
                ]
            }

    async def synthesize(text):
        return text

    async def play(audio, on_start):
        on_start()
        played.append(audio)
        heard_first.set()

    session = VoiceSession(Graph(), {"messages": []}, lambda _: "Docker?", synthesize, play, print)
    async with asyncio.timeout(2):
        turn = session.start_utterance()
        session.submit(turn, object(), 0.3)
        await session.wait()
        await session.close()

    assert played == ["Containers share the host kernel.", "They isolate each application."]


async def test_docker_is_interrupted_and_kubernetes_answers_via_compiled_graph(monkeypatch, caplog):
    import json
    import logging
    import time

    from langchain_core.messages import AIMessageChunk

    from isaac.config.settings import get_settings
    from isaac.core import graph
    from isaac.core.state import make_initial_state

    monkeypatch.setattr(get_settings(), "identity_profile", "selene")
    monkeypatch.setattr(graph, "guard_node", lambda _: {"guard_blocked": False})
    monkeypatch.setattr(graph, "meta_learner_node", lambda _: {"current_phase": "meta_learner"})
    playing = asyncio.Event()
    stopped = asyncio.Event()
    model_closed = asyncio.Event()
    played = []
    questions = []
    reports = []

    class Model:
        async def astream(self, prompt):
            assert "English only" in prompt[0].content
            assert "brief spoken answer" in prompt[0].content
            question = prompt[-1].content
            questions.append(question)
            if "Docker" in question:
                try:
                    yield AIMessageChunk(
                        content="", additional_kwargs={"reasoning_content": "private"}
                    )
                    yield AIMessageChunk(content="Docker containers share the host kernel.")
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    # A provider may still yield a late chunk when its request is canceled.
                    yield AIMessageChunk(content="Stale Docker audio.")
                finally:
                    model_closed.set()
            else:
                yield AIMessageChunk(content="Kubernetes orchestrates containers.")

    monkeypatch.setattr("isaac.llm.provider.get_direct_response_llm", Model)

    async def synthesize(text):
        return text

    async def play(audio, on_start):
        on_start()
        played.append(audio)
        if "Docker" in audio:
            playing.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    state = dict(make_initial_state())
    state.update(response_language="English only", voice_session=True)
    session = VoiceSession(graph.build_graph(), state, str, synthesize, play, reports.append)
    caplog.set_level(logging.INFO, logger="isaac.interfaces.realtime_voice")

    async with asyncio.timeout(3):
        try:
            previous = session.start_utterance()
            session.submit(previous, "Selene, explain how Docker containers work.", 0.6)
            await playing.wait()
            assert not model_closed.is_set()
            current = session.start_utterance(time.monotonic())
            await stopped.wait()
            session.submit(current, "Selene, stop. What is Kubernetes?", 0.6)
            await session.wait()
        finally:
            await session.close()

    assert model_closed.is_set()
    assert played == [
        "Docker containers share the host kernel.",
        "Kubernetes orchestrates containers.",
    ]
    assert questions == ["SELENE, explain how Docker containers work.", "What is Kubernetes?"]
    assert [message.content for message in session.state["messages"]] == [
        "What is Kubernetes?",
        "Kubernetes orchestrates containers.",
    ]
    assert not any("[ERROR]" in line for line in reports)
    metrics = [json.loads(line) for line in caplog.messages if line.startswith('{"event"')]
    assert {event["event"] for event in metrics} == {
        "speech_endpoint",
        "stt_transcription",
        "response_latency",
        "tts_first_audio_ready",
        "first_llm_token",
        "first_tts_segment",
        "reply_text_ready",
        "first_spoken_audio",
        "barge_in_cancel",
    }
    assert {event["turn_id"] for event in metrics} == {previous.id, current.id}
    assert all(event["latency_ms"] >= 0 for event in metrics)


async def test_superseded_tool_node_finishes_without_starting_another_node(monkeypatch):
    import threading

    from isaac.core import graph
    from isaac.core.state import PlanStep, make_initial_state

    tool_started = asyncio.Event()
    release_tool = threading.Event()
    executions = []
    reports = []
    loop = asyncio.get_running_loop()

    def connector(_):
        executions.append("started")
        loop.call_soon_threadsafe(tool_started.set)
        assert release_tool.wait(3)
        executions.append("finished")
        return {"current_phase": "connector_execution"}

    def guard(state):
        if state["messages"][-1].content == "Newest request":
            return {"guard_blocked": True, "messages": [AIMessage(content="The latest answer.")]}
        return {"guard_blocked": False}

    monkeypatch.setattr(graph, "guard_node", guard)
    monkeypatch.setattr(graph, "perception_node", lambda _: {"task_mode": "code"})
    monkeypatch.setattr(graph, "clarification_node", lambda _: {"needs_clarification": False})
    monkeypatch.setattr(graph, "explorer_node", lambda _: {"current_phase": "explorer"})
    monkeypatch.setattr(
        graph,
        "planner_node",
        lambda _: {"plan": [PlanStep(id="1", description="test", status="active")]},
    )
    monkeypatch.setattr(graph, "connector_execution_node", connector)
    monkeypatch.setattr(
        graph, "synthesis_node", lambda _: executions.append("unexpected synthesis")
    )

    async def no_playback(*_):
        pytest.fail("audio is disabled")

    session = VoiceSession(
        graph.build_graph(), dict(make_initial_state()), str, None, no_playback, reports.append
    )
    async with asyncio.timeout(3):
        try:
            first = session.start_utterance()
            session.submit(first, "Run the existing operation", None)
            await tool_started.wait()
            for superseded in ("Second request", "Third request"):
                turn = session.start_utterance()
                session.submit(turn, superseded, None)
            latest = session.start_utterance()
            session.submit(latest, "Newest request", None)
            # A stale endpoint must never replace the latest pending utterance.
            session.submit(first, "Stale speech", None)
            assert executions == ["started"]
            release_tool.set()
            await session.wait()
        finally:
            release_tool.set()
            await session.close()

    assert executions == ["started", "finished"]
    assert reports == [
        ">>> Run the existing operation",
        ">>> Newest request",
        "[SELENE] The latest answer.\n",
    ]


async def test_interruption_cancels_blocked_synthesis_and_bounds_queued_sentences():
    synth_started = asyncio.Event()
    synth_cancelled = asyncio.Event()
    enqueued = []
    played = []

    class Graph:
        async def astream(self, state, *, stream_mode):
            turn = current_voice_turn.get()
            if state["messages"][-1].content == "Long question":
                with turn.generation():
                    for i in range(100):
                        await turn.token(f"Old sentence number {i}. ")
                        enqueued.append(i)
            yield {"messages": [*state["messages"], AIMessage(content="New answer.")]}

    async def synthesize(text):
        if text.startswith("Old"):
            synth_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                synth_cancelled.set()
                return "late audio"
        return text

    async def play(audio, on_start):
        on_start()
        played.append(audio)

    tasks_before = asyncio.all_tasks()
    session = VoiceSession(Graph(), {"messages": []}, str, synthesize, play, print)
    async with asyncio.timeout(2):
        try:
            old = session.start_utterance()
            session.submit(old, "Long question", None)
            await synth_started.wait()
            await asyncio.sleep(0)
            assert 1 <= len(enqueued) < 10
            replacement = session.start_utterance()
            session.submit(replacement, "Next question", None)
            await session.wait()
        finally:
            await session.close()

    assert synth_cancelled.is_set()
    assert played == ["New answer."]
    assert asyncio.all_tasks() == tasks_before


async def test_recovers_from_mic_disconnect_and_empty_transcript():
    import time

    from isaac.interfaces.realtime_voice import listen_hands_free
    from isaac.multimodal.voice.audio_io import CaptureEvent

    opens = 0
    transcribed = []
    reports = []

    async def microphone():
        nonlocal opens
        opens += 1
        yield CaptureEvent(time.monotonic())
        if opens == 1:
            raise OSError("headset disconnected")
        yield CaptureEvent(time.monotonic(), audio="empty")
        await session.wait()
        yield CaptureEvent(time.monotonic())
        yield CaptureEvent(time.monotonic(), audio="quit")
        await asyncio.Event().wait()

    def transcribe(audio):
        transcribed.append(audio)
        return "" if audio == "empty" else audio

    session = VoiceSession(None, {}, transcribe, None, None, reports.append)
    tasks_before = asyncio.all_tasks()
    async with asyncio.timeout(2):
        await listen_hands_free(session, microphone, retry_s=0)

    assert opens == 2
    assert transcribed == ["empty", "quit"]
    assert reports == [
        "Microphone error: headset disconnected — retrying.",
        "(no speech recognised)",
        ">>> quit",
    ]
    assert asyncio.all_tasks() == tasks_before


async def test_recovers_from_a_model_timeout_and_a_playback_failure(monkeypatch):
    from langchain_core.messages import AIMessageChunk

    from isaac.config.settings import get_settings
    from isaac.core import graph
    from isaac.core.state import make_initial_state

    monkeypatch.setattr(get_settings(), "voice_llm_token_timeout_s", 0.01)
    monkeypatch.setattr(graph, "guard_node", lambda _: {"guard_blocked": False})
    monkeypatch.setattr(graph, "meta_learner_node", lambda _: {"current_phase": "meta_learner"})
    reports = []
    played = []

    class Model:
        async def astream(self, prompt):
            if "slow" in prompt[-1].content:
                await asyncio.Event().wait()
            yield AIMessageChunk(content=f"Answer to {prompt[-1].content}")

    async def synthesize(text):
        return text

    async def play(audio, on_start):
        if "broken" in audio:
            raise OSError("speaker disconnected")
        on_start()
        played.append(audio)

    monkeypatch.setattr("isaac.llm.provider.get_direct_response_llm", Model)
    state = dict(make_initial_state())
    state.update(response_language="English only", voice_session=True)
    session = VoiceSession(graph.build_graph(), state, str, synthesize, play, reports.append)
    async with asyncio.timeout(3):
        try:
            for topic in ("slow", "broken", "working"):
                turn = session.start_utterance()
                session.submit(turn, f"What is {topic}?", None)
                await session.wait()
        finally:
            await session.close()

    assert "[ERROR] Timed out waiting for a model response." in reports
    assert "Playback failed: speaker disconnected" in reports
    assert played == ["Answer to What is working?"]


async def test_two_consecutive_audible_replies_can_be_interrupted():
    played = asyncio.Queue(maxsize=1)
    aborted = []

    class Graph:
        async def astream(self, state, *, stream_mode):
            reply = f"Answer {state['messages'][-1].content}."
            turn = current_voice_turn.get()
            with turn.generation():
                await turn.token(reply)
                if reply != "Answer three.":
                    await asyncio.Event().wait()
            yield {"messages": [*state["messages"], AIMessage(content=reply)]}

    async def synthesize(text):
        return text

    async def play(audio, on_start):
        on_start()
        await played.put(audio)
        if audio != "Answer three.":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                aborted.append(audio)
                raise

    tasks_before = asyncio.all_tasks()
    session = VoiceSession(Graph(), {"messages": []}, str, synthesize, play, print)
    async with asyncio.timeout(2):
        try:
            for question in ("one", "two", "three"):
                turn = session.start_utterance()
                session.submit(turn, question, None)
                assert await played.get() == f"Answer {question}."
            await session.wait()
        finally:
            await session.close()

    assert aborted == ["Answer one.", "Answer two."]
    assert asyncio.all_tasks() == tasks_before


@pytest.mark.parametrize(
    ("question", "answer", "model_calls"),
    [
        ("can you hear me?", "Yes, I can hear you.", 0),
        ("what is your name?", "My name is SELENE.", 0),
        ("explain what Python is.", "Python is a programming language.", 1),
        (
            "explain neural networks in two sentences.",
            "They learn patterns. Layers transform inputs.",
            1,
        ),
    ],
)
async def test_conversation_uses_direct_voice_route(monkeypatch, question, answer, model_calls):
    from langchain_core.messages import AIMessageChunk

    from isaac.config.settings import get_settings
    from isaac.core import graph
    from isaac.core.state import make_initial_state

    monkeypatch.setattr(get_settings(), "identity_profile", "selene")
    monkeypatch.setattr(graph, "guard_node", lambda _: {"guard_blocked": False})

    def unexpected(*args, **kwargs):
        pytest.fail("Conversational request reached classification or exploration")

    monkeypatch.setattr("isaac.llm.provider.get_perception_llm", unexpected)
    monkeypatch.setattr(graph, "explorer_node", unexpected)
    calls = []

    class Model:
        async def astream(self, prompt):
            assert "English only" in prompt[0].content
            calls.append(prompt)
            yield AIMessageChunk(content=answer)

    monkeypatch.setattr("isaac.llm.provider.get_direct_response_llm", Model)
    spoken = []

    async def synthesize(text):
        return text

    async def play(segment, on_start):
        on_start()
        spoken.append(segment)

    state = dict(make_initial_state(), response_language="English only", voice_session=True)
    session = VoiceSession(graph.build_graph(), state, str, synthesize, play, lambda _: None)
    session.submit(session.start_utterance(), "Selene, " + question, 0.3)
    await asyncio.wait_for(session.wait(), 3)
    assert " ".join(spoken) == answer
    assert len(calls) == model_calls
