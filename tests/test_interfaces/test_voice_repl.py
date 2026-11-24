"""Integration tests for the SELENE voice turn with simulated external I/O."""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from isaac.config.settings import get_settings
from isaac.interfaces import voice_repl
from isaac.multimodal import voice
from isaac.multimodal.voice import audio_io, stt, tts


@pytest.mark.parametrize("tts_status", ["available", "missing", "network_error"])
def test_voice_turn_routes_speech(monkeypatch, capsys, tts_status):
    monkeypatch.setattr(get_settings(), "identity_profile", "selene")
    raw_question = "Celene, what is PipeWire?"
    question = "SELENE, what is PipeWire?"
    answer = "PipeWire manages audio on Linux."
    samples = np.zeros(1600, dtype=np.float32)
    recorded_paths = []

    def transcribe(path: str) -> str:
        recorded_paths.append(Path(path))
        with wave.open(path, "rb") as wav:
            assert wav.getframerate() == 16000
            assert wav.getnchannels() == 1
            assert wav.getnframes() == len(samples)
        return raw_question

    def stream(state):
        assert state["messages"][-1] == HumanMessage(content=question)
        assert state["response_language"] == "English only"
        yield {"direct_response": {"messages": [AIMessage(content=answer)]}}

    capture = Mock(return_value=samples)
    compiled = Mock(stream=Mock(side_effect=stream))
    playback = AsyncMock(
        side_effect=[OSError("offline"), None] if tts_status == "network_error" else None
    )
    inputs = iter(["", "", "/q"] if tts_status == "network_error" else ["", "/q"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(audio_io, "is_audio_available", lambda: True)
    monkeypatch.setattr(audio_io, "record_until_silence", capture)
    monkeypatch.setattr(voice, "is_stt_available", lambda: True)
    monkeypatch.setattr(voice, "is_tts_available", lambda: tts_status != "missing")
    monkeypatch.setattr(stt, "get_stt", lambda: Mock(transcribe=transcribe))
    monkeypatch.setattr(tts, "get_tts", lambda: Mock(speak=playback))
    monkeypatch.setattr(voice_repl, "build_graph", lambda: compiled)
    monkeypatch.setattr("isaac.tools.register_all_tools", Mock())
    monkeypatch.setattr("isaac.nodes.computer_use.shutdown_ui_executor", Mock())

    assert voice_repl.run_voice_repl() == 0

    output = capsys.readouterr().out
    assert f">>> {raw_question}" in output
    assert question in output
    assert "[SELENE] Voice REPL" in output
    assert f"[SELENE] {answer}" in output
    assert answer in output
    assert all(not path.exists() for path in recorded_paths)
    if tts_status == "missing":
        playback.assert_not_awaited()
        assert "text-only" in output
    elif tts_status == "network_error":
        assert "TTS failed: offline" in output
        assert capture.call_count == compiled.stream.call_count == playback.await_count == 2
    else:
        playback.assert_awaited_once_with(answer)
