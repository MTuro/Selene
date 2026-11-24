"""Regression tests for microphone transients in SELENE voice capture."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from isaac.multimodal.voice.audio_io import record_until_silence


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
