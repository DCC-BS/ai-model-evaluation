"""Pipeline for any model served behind the standard OpenAI /v1/audio/transcriptions API.

Works with: Voxtral, Qwen ASR, IBM Granite, and any other OpenAI-compatible transcription endpoint.
"""

from __future__ import annotations

import re

import openai

from asr_eval.base import ASRPipeline


class OpenAIAPIPipeline(ASRPipeline):
    """Transcribe via ``openai.audio.transcriptions.create``."""

    def __init__(
        self,
        model: str,
        api_url: str = "http://localhost:8000/v1",
        data_dir: str = "./data",
        language: str = "de",
        post_process: str | None = None,
        temperature: str | float | None = None,
        response_format: str | None = None,
    ):
        super().__init__(model=model, api_url=api_url, data_dir=data_dir)
        self.language = language
        self.post_process = post_process
        self.temperature = float(temperature) if temperature is not None else None
        self.response_format = response_format
        self._client = openai.OpenAI(base_url=self.api_url, api_key="none")

    async def transcribe(self, audio_path: str) -> str:
        extra: dict = {}
        if self.temperature is not None:
            extra["temperature"] = self.temperature
        if self.response_format is not None:
            extra["response_format"] = self.response_format

        with open(audio_path, "rb") as f:
            transcription = self._client.audio.transcriptions.create(
                file=f,
                model=self.model,
                language=self.language,
                **extra,
            )
        raw_text = transcription.text
        text = raw_text

        if self.post_process == "qwen_asr":
            text = text.replace("language German<asr_text>", " ")
            text = text.replace("</asr_text>", " ")
            text = text.strip()
        elif self.post_process == "moss":
            text = self._parse_moss(text)

        if not text.strip() and raw_text.strip():
            print(
                f"WARNING: post_process={self.post_process!r} produced an "
                f"empty transcript for {audio_path} from non-empty raw "
                f"output ({len(raw_text)} chars). Raw: {raw_text[:200]!r}"
            )

        return text

    @staticmethod
    def _parse_moss(text: str) -> str:
        """Strip MOSS-Transcribe-Diarize's ``[start][Sxx]text[end]`` markers.

        Segments are just concatenated with no separator. The ``[Sxx]``
        speaker tag isn't always present, so rather than anchoring on it
        (which silently dropped speaker-less segments -> empty
        transcriptions), strip every bracketed token — timestamp or speaker
        tag alike — and keep whatever text is left.
        """
        return re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]", " ", text)).strip()
