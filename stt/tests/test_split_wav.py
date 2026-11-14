import io
import wave

import numpy as np

from asr_eval.pipelines.nemotron_transcribe import split_wav

SR = 16000


def write_wav(path, samples):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.astype("<i2").tobytes())


def frames(piece):
    return wave.open(io.BytesIO(piece)).getnframes()


def test_short_file_is_one_piece(tmp_path):
    rng = np.random.default_rng(0)
    write_wav(tmp_path / "a.wav", rng.integers(-3000, 3000, 10 * SR))
    pieces = split_wav(str(tmp_path / "a.wav"), 30)
    assert [frames(p) for p in pieces] == [10 * SR]


def test_cuts_in_the_pause_and_keeps_every_sample(tmp_path):
    rng = np.random.default_rng(0)
    audio = rng.integers(-3000, 3000, 70 * SR)
    audio[27 * SR : 28 * SR] = 0  # pause inside the search window before 30 s
    write_wav(tmp_path / "a.wav", audio)

    pieces = split_wav(str(tmp_path / "a.wav"), 30)

    lengths = [frames(p) for p in pieces]
    assert sum(lengths) == len(audio)
    assert max(lengths) <= 30 * SR
    assert 27 * SR <= lengths[0] <= 28 * SR
