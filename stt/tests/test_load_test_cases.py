from asr_eval.base import ASRPipeline


class _Stub(ASRPipeline):
    async def transcribe(self, audio_path: str) -> str:
        return ""


def test_load_test_cases_picks_complete_cases_only(tmp_path):
    complete = tmp_path / "case_001"
    complete.mkdir()
    (complete / "RecordedAudio.wav").write_bytes(b"")
    (complete / "ground_truth_transcript.txt").write_text("hallo welt\n", encoding="utf-8")

    incomplete = tmp_path / "case_002"
    incomplete.mkdir()
    (incomplete / "RecordedAudio.wav").write_bytes(b"")

    cases = _Stub(model="m", api_url="http://x", data_dir=str(tmp_path)).load_test_cases()

    assert len(cases) == 1
    assert cases[0]["audio_path"] == str(complete / "RecordedAudio.wav")
    assert cases[0]["ground_truth"] == "hallo welt"
