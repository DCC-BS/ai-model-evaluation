"""Pipeline for Qwen3-ASR models over chat completions (vLLM).

Some Qwen3-ASR fine-tunes take their transcription options through the chat
template, which vLLM's ``/v1/audio/transcriptions`` route does not use. For
example, Reson8/Rumble-2-Flash reads ``language``, ``digits``, ``filler`` and
``verbatim`` from ``chat_template_kwargs``. This pipeline sends the audio
alone in the user message and passes those options through.

Options that are left unset are not sent, so the model's template defaults
apply. ``chunk_s`` cuts long files into pieces, as in ``nemotron_transcribe``.
"""

from __future__ import annotations

import asyncio
import base64
import re

import httpx

from asr_eval.audio import split_wav
from asr_eval.base import ASRPipeline

# Without a forced language the model writes "language German<asr_text>..."
_LANGUAGE_PREFIX = re.compile(r"language\s+\w+\s*<asr_text>")
_SPECIAL = re.compile(r"</?asr_text>")


def _flag(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes")


class Qwen3ASRChatPipeline(ASRPipeline):
    """Transcribe via ``/v1/chat/completions`` with template options."""

    def __init__(
        self,
        model: str,
        api_url: str = "http://localhost:8000/v1",
        data_dir: str = "./data",
        asr_language: str | None = "German",
        digits: str | bool | None = None,
        filler: str | bool | None = None,
        verbatim: str | bool | None = None,
        chunk_s: str | float = 0,
        max_tokens: str | int = 4096,
        timeout: str | float = 3600,
        **kwargs,
    ):
        # Suites pass ``language: de`` to every run; Qwen3-ASR wants the
        # English language name instead, so it has its own ``asr_language``.
        super().__init__(model=model, api_url=api_url, data_dir=data_dir, **kwargs)
        options = {
            "language": asr_language or None,
            "digits": _flag(digits),
            "filler": _flag(filler),
            "verbatim": _flag(verbatim),
        }
        self.template_kwargs = {k: v for k, v in options.items() if v is not None}
        self.chunk_s = float(chunk_s)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)

    async def transcribe(self, audio_path: str) -> str:
        if self.chunk_s <= 0:
            with open(audio_path, "rb") as f:
                return await self._transcribe_wav(f.read(), audio_path)
        chunks = split_wav(audio_path, self.chunk_s)
        texts = await asyncio.gather(
            *(self._transcribe_wav(c, f"{audio_path}#{i}") for i, c in enumerate(chunks))
        )
        return re.sub(r"\s+", " ", " ".join(texts)).strip()

    async def _transcribe_wav(self, wav_bytes: bytes, label: str) -> str:
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                        }
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": self.template_kwargs,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.api_url}/chat/completions", json=payload)
            response.raise_for_status()
            choice = response.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            print(f"WARNING: hit max_tokens={self.max_tokens} for {label}")
        return self.clean(choice["message"].get("content") or "")

    @staticmethod
    def clean(text: str) -> str:
        text = _LANGUAGE_PREFIX.sub(" ", text)
        text = _SPECIAL.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip()
