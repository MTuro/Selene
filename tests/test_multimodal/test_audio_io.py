"""Regression tests for microphone transients in SELENE voice capture."""

from __future__ import annotations

import asyncio
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from isaac.config.settings import Settings
from isaac.multimodal.voice.audio_io import UtteranceDetector, play_pcm, record_until_silence


@pytest.mark.parametrize("speech", [False, True])
def test_capture_ignores_noise_before_speech(monkeypatch, speech):
    amplitudes = [0.25] * 3 + [0.0] * 45
    if speech:
        amplitudes += [0.1] * 8 + [0.0] * 40
    stream = MagicMock()
    stream.__enter__.return_value.read.side_effect = [
        (np.full((480, 1), amplitude, dtype=np.float32), False) for amplitude in amplitudes
    ]
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=lambda **_: stream))
    monkeypatch.setitem(sys.modules, "webrtcvad", None)

    audio = record_until_silence(max_record_s=len(amplitudes) * 0.03)

    if speech:
        assert audio.size == (10 + 3 + 40) * 480
        np.testing.assert_allclose(audio.max(), 0.1)
    else:
        assert audio.size == 0


def test_barge_in_vad_rejects_headset_leakage_and_transient_noise():
    detector = UtteranceDetector(
        is_speech=lambda frame: bool(frame[0] > 0),
        energy=0.005,
        playback_energy=0.025,
        onset_frames=3,
        playback_onset_frames=4,
        silence_s=0.12,
    )
    at = time.monotonic() - 1

    def feed(level, *, playback=True):
        nonlocal at
        at += 0.03
        return detector.feed(np.full(480, level, dtype=np.float32), at, playback=playback)

    for _ in range(10):
        assert feed(0.01) is None  # voiced leakage below the playback threshold
    assert feed(0.1) is None  # loud click
    for _ in range(6):
        assert feed(-0.08) is None  # loud but rejected by WebRTC
    for _ in range(3):
        assert feed(0.08) is None
    onset = feed(0.08)
    assert onset is not None and onset.audio is None

    endpoint = None
    for _ in range(6):
        if event := feed(0, playback=False):
            endpoint = event
            break
    assert endpoint is not None
    assert endpoint.endpoint_latency_s is not None and endpoint.endpoint_latency_s >= 0.12
    assert endpoint.audio is not None
    np.testing.assert_allclose(endpoint.audio[2 * 480 : 6 * 480], 0.08)


async def test_cancelling_playback_aborts_portaudio_instead_of_draining(monkeypatch):
    started = asyncio.Event()
    events = []

    class OutputStream:
        def __init__(self, *, callback, **kwargs):
            self.callback = callback

        def start(self):
            self.callback(np.zeros((4, 1), dtype=np.float32), 4, None, None)

        def abort(self):
            events.append("aborted")

        def close(self):
            events.append("closed")

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=OutputStream))
    task = asyncio.create_task(play_pcm((np.ones((16, 1), dtype=np.float32), 16000), started.set))
    async with asyncio.timeout(1):
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert events == ["aborted", "closed"]


async def test_portaudio_finished_callback_returns_none(monkeypatch):
    class CallbackStop(Exception):
        pass

    class OutputStream:
        def __init__(self, *, callback, finished_callback, **kwargs):
            self.callback = callback
            self.finished_callback = finished_callback

        def start(self):
            with pytest.raises(CallbackStop):
                self.callback(np.zeros((8, 1), dtype=np.float32), 8, None, None)
            # PortAudio calls a C void function, which requires a None result.
            assert self.finished_callback() is None

        def abort(self):
            pytest.fail("completed audio should drain")

        def close(self):
            pass

    backend = SimpleNamespace(OutputStream=OutputStream, CallbackStop=CallbackStop)
    monkeypatch.setitem(sys.modules, "sounddevice", backend)
    await play_pcm((np.ones((8, 1), dtype=np.float32), 16000), lambda: None)


def test_barge_in_accepts_quiet_speech_with_one_gap_at_default_sensitivity():
    # Use the declared defaults independently of dotenv and shell overrides.
    settings = Settings.model_construct()
    detector = UtteranceDetector(
        is_speech=lambda frame: bool(frame[0] > 0),
        energy=settings.voice_vad_energy,
        playback_energy=settings.voice_barge_in_energy,
        onset_frames=settings.voice_vad_onset_frames,
        playback_onset_frames=settings.voice_barge_in_onset_frames,
        silence_s=settings.voice_vad_silence_s,
    )
    at = time.monotonic()

    def feed(level):
        nonlocal at
        at += 0.03
        frame = np.full(480, level, dtype=np.float32)
        return detector.feed(frame, at, playback=True)

    for _ in range(12):
        assert feed(0.009) is None
    for level in (0.016, 0.014, 0, 0.018):
        assert feed(level) is None
    onset = feed(0.015)
    assert onset is not None and onset.audio is None
