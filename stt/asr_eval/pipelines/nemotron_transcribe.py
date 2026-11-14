"""Pipeline for NVIDIA Nemotron 3.5 Transcribe (chat-completions with base64 audio).

The model is a speech LLM served by vLLM through NeMo's ``nemo_speechlm``
plugin. It has no ``/v1/audio/transcriptions`` route; the task is chosen by
the instruction in the user message. Prompts and decoding settings follow
the model card: no system message, reasoning off, greedy decoding, and the
generation-loop guardrail applied before scoring.
"""

from __future__ import annotations

import asyncio
import base64
import io
import re
import wave

import httpx
import numpy as np

from asr_eval.base import ASRPipeline

# Model-card prompts, selectable with ``--arg prompt=<name>``. Any other
# value is sent to the model as-is.
PROMPTS = {
    # "Verbatim Transcription" benchmark prompt.
    "verbatim": (
        "Audio transcription. Language: as spoken. Wording: verbatim, including "
        "fillers and repetitions. Formatting: punctuation and capitalization. "
        "Output: transcript only."
    ),
    # "Clean written transcript", with numbers in written form.
    "clean": (
        "Audio transcription. Language: as spoken. Wording: omit fillers and "
        "accidental repetitions; keep the remaining spoken wording unchanged. "
        "Formatting: punctuation and capitalization, numbers in their standard "
        "written form. Output: transcript only."
    ),
    # Speech translation into Standard German, for Swiss German recordings.
    "translate_de": "Please translate the given speech to German.",
    # Our own prompt: Swiss German speech, Standard German text, digits.
    "swiss_de": (
        "Audio transcription. The speaker talks Swiss German; write the transcript "
        "in Standard German. Wording: keep the spoken wording and order. "
        "Formatting: punctuation and capitalization; write numbers, dates and "
        "times as digits. Output: transcript only."
    ),
    # Training prompt of Flurin17/nemotron-3.5-transcribe-swiss-german.
    "swiss_ft": (
        "Transcribe this Swiss German speech into Swiss Standard German. Preserve "
        "the meaning and use Swiss spelling, punctuation and capitalization. "
        "Output only the transcript."
    ),
}

# Generation-loop guardrail from the model card (adapted from Qwen3-ASR,
# Apache-2.0).
_CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
_BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)
_SPECIAL_TOKEN = re.compile(r"<\|[^|>]*\|>|<spk:\d+>")


class NemotronTranscribePipeline(ASRPipeline):
    """Transcribe via ``/v1/chat/completions`` with the audio in the user message."""

    def __init__(
        self,
        model: str = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8",
        api_url: str = "http://localhost:8000/v1",
        data_dir: str = "./data",
        prompt: str = "clean",
        max_tokens: str | int = 8192,
        timeout: str | float = 3600,
        chunk_s: str | float = 0,
        **kwargs,
    ):
        # Tolerate shared kwargs (e.g. ``language``) set in a suite's defaults;
        # the prompt already says which language to write.
        super().__init__(model=model, api_url=api_url, data_dir=data_dir, **kwargs)
        self.prompt = PROMPTS.get(prompt, prompt)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        # 0 sends each file whole. A positive value cuts the audio into pieces
        # of at most that many seconds, for models trained on short clips.
        self.chunk_s = float(chunk_s)

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
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.api_url}/chat/completions", json=payload)
            response.raise_for_status()
            choice = response.json()["choices"][0]

        if choice.get("finish_reason") == "length":
            print(f"WARNING: hit max_tokens={self.max_tokens} for {label}")
        return self._clean(choice["message"].get("content") or "")

    @staticmethod
    def _clean(text: str) -> str:
        text = _CHAR_RUN.sub(lambda m: m.group(1), text)
        text = _BLOCK_RUN.sub(lambda m: m.group(1), text)
        text = _SPECIAL_TOKEN.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip()


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
