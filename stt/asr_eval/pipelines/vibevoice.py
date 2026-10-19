"""Pipeline for Microsoft VibeVoice (chat-completions with base64 audio)."""

from __future__ import annotations

import base64
import json
import re
import wave

import httpx

from asr_eval.base import ASRPipeline


class VibeVoicePipeline(ASRPipeline):
    """Transcribe via VibeVoice's ``/v1/chat/completions`` streaming endpoint."""

    def __init__(
        self,
        model: str = "vibevoice",
        api_url: str = "http://localhost:8000/v1",
        data_dir: str = "./data",
        **kwargs,
    ):
        # Tolerate shared kwargs (e.g. ``language``) set in a suite's defaults;
        # the base class ignores ones this pipeline doesn't use.
        super().__init__(model=model, api_url=api_url, data_dir=data_dir, **kwargs)

    async def transcribe(self, audio_path: str) -> str:
        duration = self._get_wav_duration(audio_path)
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that transcribes audio input into text output in JSON format.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"This is a {duration:.2f} seconds audio, please transcribe it "
                                "with these keys: Start time, End time, Speaker ID, Content"
                            ),
                        },
                    ],
                },
            ],
            "max_tokens": 30536,
            "temperature": 0.0,
            "stream": True,
            "top_p": 1.0,
        }

        accumulated = ""
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", f"{self.api_url}/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    json_str = line[6:]
                    if json_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(json_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            if delta.startswith(accumulated):
                                accumulated = delta
                            else:
                                accumulated += delta
                    except (json.JSONDecodeError, KeyError):
                        pass

        return self._extract_text(accumulated)

    # ------------------------------------------------------------------

    @staticmethod
    def _get_wav_duration(audio_path: str) -> float:
        with wave.open(audio_path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())

    @staticmethod
    def _extract_text(content: str) -> str:
        texts = [m.group(1) for m in re.finditer(r'"Content"\s*:\s*"([^"]*)"', content)]
        if texts:
            return " ".join(texts).strip()
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return " ".join(seg.get("Content", "") for seg in parsed).strip()
            if isinstance(parsed, dict):
                return parsed.get("Content", content).strip()
        except json.JSONDecodeError:
            pass
        return content.strip()
