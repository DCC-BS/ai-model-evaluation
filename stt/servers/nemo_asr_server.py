"""Minimal OpenAI-compatible transcription shim for NVIDIA NeMo ASR models.

NeMo models (parakeet, nemotron-asr) have no native vLLM serving, so this tiny
FastAPI app wraps ``nemo_asr.models.ASRModel.transcribe`` behind the same two
endpoints the suite needs:

  GET  /v1/models                -> readiness probe (200 once the model loaded)
  POST /v1/audio/transcriptions  -> OpenAI-shaped {"text": ...}

The model id is taken from the ``NEMO_MODEL`` env var so one server file serves
any NeMo ASR checkpoint. Audio is downmixed to mono and resampled to 16 kHz
(what these models expect) before transcription.
"""

from __future__ import annotations

import io
import os
import re
import tempfile

import librosa
import nemo.collections.asr as nemo_asr
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

MODEL_NAME = os.environ["NEMO_MODEL"]
TARGET_SR = 16000

app = FastAPI()
_model = None

# Set True for prompt-conditioned models (nemotron-3.5-asr): they need a target
# language prompt and emit a trailing <lang> tag. Plain models leave this False.
_is_prompt_model = False
# Language forced into the next transcribe() call. Runs are single-language
# (one `--arg language=...` per suite run) and the base pipeline reuses one
# value for every concurrent request, so a module global is race-free here.
_forced_lang = "auto"

# Language tag nemotron appends after each utterance's terminal punctuation,
# e.g. "... am Montag. <de-DE> Nächster ...". The whole file is transcribed
# as one string, so several can appear inline; strip them all.
_LANG_TAG_RE = re.compile(r"\s*<[A-Za-z]{2,3}(?:-[A-Za-z]{2})?>\s*")


def _install_prompt_patch() -> None:
    # nemotron's offline transcribe() builds a Lhotse dataloader whose temp
    # manifest carries no language, so the prompt-index dataset reads
    # cut.supervisions[0].language == None and raises "Unknown prompt key".
    # (target_lang / default_lang never reach the cut.) Override the per-cut
    # prompt selection to use the request's language, or the model's own auto
    # language-ID index when unset/"auto".
    from nemo.collections.asr.data import audio_to_text_lhotse_prompt_index as p

    cls = p.LhotseSpeechToTextBpeDatasetWithPromptIndex

    def _prompt_index_for_cut(self, cut):  # noqa: ANN001, ARG001
        if _forced_lang and _forced_lang != "auto":
            return self._get_prompt_index(_forced_lang)
        return self.auto_index

    cls._get_prompt_index_for_cut = _prompt_index_for_cut


@app.on_event("startup")
def _load() -> None:
    # Loads before uvicorn starts serving, so /v1/models stays unreachable
    # until the checkpoint is ready -- exactly the readiness signal the suite
    # polls for.
    global _model, _is_prompt_model
    print(f"Loading NeMo ASR model: {MODEL_NAME}", flush=True)
    _model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
    _model.eval()
    _is_prompt_model = hasattr(_model, "num_prompts")
    if _is_prompt_model:
        _install_prompt_patch()
    print("Model ready", flush=True)


@app.get("/v1/models")
def list_models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(None),  # noqa: ARG001 -- accepted for OpenAI compat, ignored
    language: str = Form(None),
) -> dict:
    raw = await file.read()
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if audio.ndim > 1:  # downmix to mono
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    # Prompt-conditioned models pick their language prompt from the OpenAI
    # `language` form field (e.g. "de", "de-DE"); "auto" / unset lets the model
    # detect it. Plain models ignore this entirely.
    if _is_prompt_model:
        global _forced_lang
        _forced_lang = language or "auto"

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, audio, TARGET_SR)
        out = _model.transcribe([tmp.name])

    hyp = out[0]
    text = hyp.text if hasattr(hyp, "text") else str(hyp)
    if _is_prompt_model:
        text = _LANG_TAG_RE.sub(" ", text).strip()
    return {"text": text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
