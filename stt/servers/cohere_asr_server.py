"""Minimal OpenAI-compatible transcription shim for Cohere Transcribe checkpoints in Transformers format.

vLLM's Cohere ASR loader expects the tensor names of the original
CohereLabs/cohere-transcribe-03-2026 checkpoint. Fine-tunes saved with
Transformers' built-in ``CohereAsrForConditionalGeneration`` use other names
and fail to load there, so this app runs them with Transformers instead,
following the model card's inference code:

  GET  /v1/models                -> readiness probe (200 once the model loaded)
  POST /v1/audio/transcriptions  -> OpenAI-shaped {"text": ...}

The model id comes from the ``HF_MODEL`` env var. The processor splits long
recordings into chunks by itself. Input must be 16-bit PCM WAV; it is
downmixed to mono and resampled to 16 kHz.
"""

from __future__ import annotations

import asyncio
import io
import os
import wave

import numpy as np
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

MODEL_NAME = os.environ["HF_MODEL"]
TARGET_SR = 16000
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))

app = FastAPI()
_processor = None
_model = None
_lock = asyncio.Lock()


@app.on_event("startup")
def _load() -> None:
    global _processor, _model
    print(f"Loading {MODEL_NAME}", flush=True)
    _processor = AutoProcessor.from_pretrained(MODEL_NAME)
    _model = CohereAsrForConditionalGeneration.from_pretrained(
        MODEL_NAME, device_map="cuda", dtype=torch.bfloat16
    )
    _model.eval()
    print("Model ready", flush=True)


@app.get("/v1/models")
def list_models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


def _read_wav(raw: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(raw), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is supported")
        sr, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    audio = pcm.reshape(-1, channels).mean(axis=1).astype(np.float32) / 32768.0
    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, TARGET_SR).numpy()
    return audio


def _transcribe(audio: np.ndarray, language: str) -> str:
    inputs = _processor(audio, sampling_rate=TARGET_SR, return_tensors="pt", language=language)
    inputs.to(_model.device, dtype=_model.dtype)
    with torch.inference_mode():
        outputs = _model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    text = _processor.decode(
        outputs,
        skip_special_tokens=True,
        audio_chunk_index=inputs.get("audio_chunk_index"),
        language=language,
    )
    return text[0].strip()


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(None),  # noqa: ARG001 -- accepted for OpenAI compat, ignored
    language: str = Form("de"),
) -> dict:
    audio = _read_wav(await file.read())
    async with _lock:
        text = await asyncio.to_thread(_transcribe, audio, language or "de")
    return {"text": text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
