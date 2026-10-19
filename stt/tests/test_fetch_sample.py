import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_sample import write_case  # noqa: E402


def test_write_case_layout(tmp_path):
    case = write_case(tmp_path, "case_001", np.zeros(16000), 16000, "hallo welt")

    assert case == tmp_path / "case_001"
    audio, rate = sf.read(case / "RecordedAudio.wav")
    assert rate == 16000
    assert len(audio) == 16000
    transcript = (case / "ground_truth_transcript.txt").read_text(encoding="utf-8")
    assert transcript == "hallo welt"
