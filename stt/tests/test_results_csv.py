import csv
import importlib

import asr_eval.base


def _reload_base(monkeypatch, results_root=None):
    if results_root is None:
        monkeypatch.delenv("AI_EVAL_RESULTS_DIR", raising=False)
    else:
        monkeypatch.setenv("AI_EVAL_RESULTS_DIR", str(results_root))
    return importlib.reload(asr_eval.base)


def test_results_dir_defaults_to_local_results(monkeypatch):
    base = _reload_base(monkeypatch)
    assert str(base.results_dir()) == "results"


def test_results_dir_follows_env(monkeypatch, tmp_path):
    base = _reload_base(monkeypatch, tmp_path)
    assert base.results_dir() == tmp_path / "stt"
    assert base.ASRPipeline.RESULTS_CSV == tmp_path / "stt" / "asr_evaluation_results.csv"


def test_old_csv_gets_new_columns(monkeypatch, tmp_path):
    base = _reload_base(monkeypatch, tmp_path)

    class Stub(base.ASRPipeline):
        async def transcribe(self, audio_path: str) -> str:
            return ""

    csv_path = tmp_path / "stt" / "asr_evaluation_results.csv"
    csv_path.parent.mkdir(parents=True)
    old_fields = ["evaluation_date", "num_test_cases", "evaluation_api", "model", "WER", "CER", "BLEU"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow({k: "old" for k in old_fields})

    Stub(model="m", api_url="http://x").write_results_to_csv(
        {"model": "m", "WER": 0.5, "pipeline": "openai_api", "args": "{}"}
    )

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["model"] == "old" and rows[0]["pipeline"] == ""
    assert rows[1]["model"] == "m" and rows[1]["pipeline"] == "openai_api"
    _reload_base(monkeypatch)
