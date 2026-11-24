"""Audio helpers shared by pipelines."""

from __future__ import annotations

import io
import wave

import numpy as np


def split_wav(audio_path: str, max_s: float, search_s: float = 5.0, frame_s: float = 0.1) -> list[bytes]:
    """Cut a 16-bit PCM WAV into pieces of at most ``max_s`` seconds.

    Each cut goes at the quietest ``frame_s`` frame (lowest RMS) in the last
    ``search_s`` seconds before the limit, so cuts land in pauses where there
    are any. Returns each piece as WAV bytes with the source's format.
    """
    with wave.open(audio_path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{audio_path}: only 16-bit PCM is supported")
        params = w.getparams()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    channels, sr = params.nchannels, params.framerate
    pcm = pcm.reshape(-1, channels)
    mono = pcm.mean(axis=1)

    max_n, search_n, frame_n = int(max_s * sr), int(search_s * sr), int(frame_s * sr)
    cuts, start = [0], 0
    while len(mono) - start > max_n:
        lo, hi = start + max_n - search_n, start + max_n
        frames = mono[lo:hi][: (hi - lo) // frame_n * frame_n].reshape(-1, frame_n)
        quietest = int(np.argmin(np.sqrt((frames**2).mean(axis=1))))
        start = lo + quietest * frame_n + frame_n // 2
        cuts.append(start)
    cuts.append(len(mono))

    pieces = []
    for a, b in zip(cuts, cuts[1:]):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setparams(params)
            out.writeframes(pcm[a:b].tobytes())
        pieces.append(buf.getvalue())
    return pieces
