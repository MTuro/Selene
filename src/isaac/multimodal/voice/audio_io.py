"""Audio I/O — microphone capture, speaker playback, and VAD-based recording.

Backends:
* ``sounddevice`` (preferred — cross-platform PortAudio)
* ``simpleaudio`` (playback only fallback)

The :func:`record_until_silence` helper implements voice-activity-detection
(VAD) using ``webrtcvad`` when available, falling back to a simple energy
gate otherwise.  This is what the voice REPL uses for push-to-talk and
hands-free conversational mode.

Modified for SELENE to reject short microphone transients before speech.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import wave
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from isaac.config.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_MS = 30
DEFAULT_SILENCE_TIMEOUT_S = 1.2
DEFAULT_MAX_RECORD_S = 30.0
_SPEECH_START_FRAMES = 5
_PRE_ROLL_FRAMES = 10


def is_audio_available() -> bool:
    """Return True if microphone capture is supported."""
    try:
        import sounddevice  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------


def play_wav(path: str | os.PathLike[str]) -> None:
    """Block-play a WAV file through the default output device."""
    path = str(path)
    try:
        import sounddevice as sd
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32")
        sd.play(data, sr)
        sd.wait()
        return
    except ImportError:
        pass

    try:
        import simpleaudio as sa  # type: ignore[import-not-found]

        wave_obj = sa.WaveObject.from_wave_file(path)
        play = wave_obj.play()
        play.wait_done()
        return
    except ImportError:
        pass

    # Last-resort: hand off to the OS
    if os.name == "nt":
        import winsound  # type: ignore[import-not-found]

        winsound.PlaySound(path, winsound.SND_FILENAME)  # type: ignore[attr-defined]
        return

    raise RuntimeError(
        "No audio playback backend available. Install one of: sounddevice+soundfile, simpleaudio."
    )


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def record_fixed(
    duration_s: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Record exactly *duration_s* seconds of mono audio."""
    import sounddevice as sd

    frames = int(duration_s * sample_rate)
    data = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return data.flatten()


def record_until_silence(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    silence_timeout_s: float = DEFAULT_SILENCE_TIMEOUT_S,
    max_record_s: float = DEFAULT_MAX_RECORD_S,
    energy_threshold: float = 0.005,
    on_chunk: Any = None,
    on_endpoint: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Record through the first phrase, returning no samples if speech never starts.

    Parameters
    ----------
    sample_rate:
        Capture sample rate.  16 kHz is the standard for STT.
    silence_timeout_s:
        How long the audio must remain below ``energy_threshold`` before
        recording stops.
    max_record_s:
        Hard ceiling on recording length.
    energy_threshold:
        RMS amplitude (0..1) below which a frame counts as silent.
        Used as the fallback when ``webrtcvad`` is unavailable.
    on_chunk:
        Optional callback ``f(rms_level: float)`` called once per frame
        — useful for drawing a level meter in the REPL.
    on_endpoint:
        Optional callback with the time from the last voiced frame to endpoint detection.
    """
    import numpy as np
    import sounddevice as sd

    frame_size = int(sample_rate * DEFAULT_FRAME_MS / 1000)
    silence_frames_needed = int(silence_timeout_s * 1000 / DEFAULT_FRAME_MS)
    max_frames = int(max_record_s * 1000 / DEFAULT_FRAME_MS)

    vad: Any = None
    try:
        import webrtcvad  # type: ignore[import-not-found]

        vad = webrtcvad.Vad(2)  # 0 = least aggressive, 3 = most
    except ImportError:
        pass

    def _is_silent(frame: np.ndarray) -> bool:
        if vad is not None:
            pcm16 = (frame * 32767).astype(np.int16).tobytes()
            try:
                return not vad.is_speech(pcm16, sample_rate)
            except Exception:
                pass  # fall through to energy gate
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        return rms < energy_threshold

    chunks: list[np.ndarray] = []
    pre_roll: deque[np.ndarray] = deque(maxlen=_PRE_ROLL_FRAMES)
    voiced_streak = 0
    silent_streak = 0
    last_voice = 0.0

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        for _ in range(max_frames):
            frame, _overflow = stream.read(frame_size)
            frame = frame.flatten()
            silent = _is_silent(frame)
            if not silent:
                last_voice = time.monotonic()
            if on_chunk is not None:
                try:
                    rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
                    on_chunk(rms)
                except Exception:
                    pass
            if not chunks:
                pre_roll.append(frame)
                voiced_streak = 0 if silent else voiced_streak + 1
                if voiced_streak >= _SPEECH_START_FRAMES:
                    chunks.extend(pre_roll)
                continue

            chunks.append(frame)
            silent_streak = silent_streak + 1 if silent else 0
            if silent and silent_streak >= silence_frames_needed:
                if on_endpoint is not None:
                    on_endpoint(time.monotonic() - last_voice)
                break

    if not chunks:
        return np.zeros(0, dtype="float32")
    return np.concatenate(chunks).astype("float32")


def save_wav(
    samples: np.ndarray,
    path: str | os.PathLike[str],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> str:
    """Write a float32 numpy array to disk as a 16-bit PCM WAV file."""
    import numpy as np

    path = str(path)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return path


def samples_to_wav_bytes(
    samples: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    """Return WAV-encoded bytes for an in-memory audio buffer."""
    import numpy as np

    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return buf.getvalue()


@dataclass(frozen=True)
class CaptureEvent:
    """A confirmed onset (audio is None) or a completed utterance."""

    at: float
    audio: np.ndarray | None = None
    endpoint_latency_s: float | None = None


def _webrtc_detector(mode: int) -> Callable[[np.ndarray], bool]:
    try:
        import webrtcvad  # type: ignore[import-not-found]
    except ImportError:
        return lambda _: True

    vad = webrtcvad.Vad(mode)

    def is_speech(frame: np.ndarray) -> bool:
        import numpy as np

        pcm = (np.clip(frame, -1, 1) * 32767).astype(np.int16).tobytes()
        return bool(vad.is_speech(pcm, DEFAULT_SAMPLE_RATE))

    return is_speech


class UtteranceDetector:
    """Confirm sustained speech and preserve its onset with bounded pre-roll."""

    def __init__(
        self,
        *,
        is_speech: Callable[[np.ndarray], bool],
        energy: float,
        playback_energy: float,
        onset_frames: int,
        playback_onset_frames: int,
        silence_s: float,
    ) -> None:
        self.is_speech = is_speech
        self.energy = energy
        self.playback_energy = playback_energy
        self.onset_frames = onset_frames
        self.playback_onset_frames = playback_onset_frames
        self.silence_s = silence_s
        self._history: deque[np.ndarray] = deque(
            maxlen=max(_PRE_ROLL_FRAMES, onset_frames + 2, playback_onset_frames + 3)
        )
        self._playback_votes: deque[bool] = deque(maxlen=playback_onset_frames + 1)
        self._frames: list[np.ndarray] = []
        self._voiced = 0
        self._last_voice = 0.0
        self._playback = False

    def feed(self, frame: np.ndarray, at: float, *, playback: bool) -> CaptureEvent | None:
        import numpy as np

        rms = float(np.sqrt(np.mean(frame * frame)))
        vad = self.is_speech(frame)
        threshold = max(self.energy, self.playback_energy) if playback else self.energy
        voiced = vad and rms >= threshold
        if not self._frames:
            if playback != self._playback:
                self._voiced = 0
                self._playback_votes.clear()
            self._playback = playback
            self._history.append(frame)
            if playback:
                # A quiet consonant or a single missed VAD frame can split real speech.
                self._playback_votes.append(voiced)
                if sum(self._playback_votes) < self.playback_onset_frames:
                    return None
                onset = len(self._playback_votes) - self._playback_votes.index(True)
                self._frames = list(self._history)[-(onset + 2) :]
            else:
                self._voiced = self._voiced + 1 if voiced else 0
                if self._voiced < self.onset_frames:
                    return None
                self._frames = list(self._history)
            self._history.clear()
            self._last_voice = at
            return CaptureEvent(at)

        self._frames.append(frame)
        if voiced:
            self._last_voice = at
        silence = at - self._last_voice >= self.silence_s
        if not silence and len(self._frames) < int(DEFAULT_MAX_RECORD_S * 1000 / DEFAULT_FRAME_MS):
            return None

        audio = np.concatenate(self._frames)
        self._frames.clear()
        self._playback_votes.clear()
        self._voiced = 0
        return CaptureEvent(at, audio, time.monotonic() - self._last_voice if silence else None)


async def capture_utterances(
    echo_gate: Callable[[], bool],
    settings: Settings,
) -> AsyncIterator[CaptureEvent]:
    """Read continuous PortAudio frames with bounded buffering and a mic watchdog."""
    import sounddevice as sd

    frame_size = DEFAULT_SAMPLE_RATE * DEFAULT_FRAME_MS // 1000
    frames: deque[tuple[float, np.ndarray]] = deque(maxlen=20)
    detector = UtteranceDetector(
        is_speech=_webrtc_detector(settings.voice_vad_mode),
        energy=settings.voice_vad_energy,
        playback_energy=settings.voice_barge_in_energy,
        onset_frames=settings.voice_vad_onset_frames,
        playback_onset_frames=settings.voice_barge_in_onset_frames,
        silence_s=settings.voice_vad_silence_s,
    )

    def receive(indata: np.ndarray, _count: int, _time: Any, _status: Any) -> None:
        frames.append((time.monotonic(), indata[:, 0].copy()))

    with sd.InputStream(
        samplerate=DEFAULT_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=frame_size,
        latency="low",
        callback=receive,
    ) as stream:
        last_frame = time.monotonic()
        while stream.active and time.monotonic() - last_frame < 1.5:
            if not frames:
                await asyncio.sleep(0.01)
                continue
            last_frame, frame = frames.popleft()
            if event := detector.feed(frame, last_frame, playback=echo_gate()):
                yield event
    raise OSError("Microphone stopped delivering audio")


async def play_pcm(
    audio: tuple[np.ndarray, int],
    on_start: Callable[[], None],
) -> None:
    """Play one decoded segment with a stream owned by this cancellable coroutine."""
    import sounddevice as sd

    samples, sample_rate = audio
    if len(samples) == 0:
        raise ValueError("TTS returned empty audio")
    loop = asyncio.get_running_loop()
    finished = asyncio.Event()
    position = 0

    def send(outdata: np.ndarray, count: int, _time: Any, _status: Any) -> None:
        nonlocal position
        if position == 0:
            loop.call_soon_threadsafe(on_start)
        end = min(position + count, len(samples))
        read = end - position
        outdata[:read] = samples[position:end]
        outdata[read:] = 0
        position = end
        if position == len(samples):
            raise sd.CallbackStop

    def on_finish() -> None:
        loop.call_soon_threadsafe(finished.set)

    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=samples.shape[1],
        dtype="float32",
        latency="low",
        callback=send,
        finished_callback=on_finish,
    )
    try:
        stream.start()
        async with asyncio.timeout(len(samples) / sample_rate + 2):
            await finished.wait()
        if position != len(samples):
            raise OSError("Speaker stopped during playback")
    finally:
        try:
            if not finished.is_set():
                stream.abort()
        finally:
            stream.close()
