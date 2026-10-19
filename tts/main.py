# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A Gradio demo for Qwen3 TTS via vLLM-omni OpenAI-compatible /v1/audio/speech API.
"""

import argparse
import base64
import io
import json
import os
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import httpx
import numpy as np
from scipy.io import wavfile


def _title_case_display(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(
    items: Optional[List[str]],
) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


# Default vLLM-omni OpenAI-compatible API configuration
DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_API_KEY = "EMPTY"


def _maybe(v):
    return v if v is not None else gr.update()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-tts-demo",
        description=(
            "Launch a Gradio demo for Qwen3 TTS via vLLM-omni OpenAI-compatible API.\n\n"
            "Examples:\n"
            "  qwen-tts-demo --api-base http://localhost:8000\n"
            "  qwen-tts-demo --api-base http://localhost:8000 --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice\n"
            "  qwen-tts-demo --api-base http://127.0.0.1:8000 --port 7860\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )

    # API configuration (vLLM-omni OpenAI-compatible /v1/audio/speech)
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help=f"API base URL (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="API key (default: EMPTY)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="Model name for the API (default: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)",
    )

    # Gradio server args
    parser.add_argument(
        "--ip",
        default="0.0.0.0",
        help="Server bind IP for Gradio (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port for Gradio (default: 8000).",
    )
    parser.add_argument(
        "--share/--no-share",
        dest="share",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to create a public Gradio link (default: disabled).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Gradio queue concurrency (default: 16).",
    )

    # HTTPS args
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Path to SSL certificate file for HTTPS (optional).",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Path to SSL key file for HTTPS (optional).",
    )
    parser.add_argument(
        "--ssl-verify/--no-ssl-verify",
        dest="ssl_verify",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to verify SSL certificate (default: enabled).",
    )

    # Optional generation args
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Max new tokens for generation (optional).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (optional).",
    )
    parser.add_argument(
        "--top-k", type=int, default=None, help="Top-k sampling (optional)."
    )
    parser.add_argument(
        "--top-p", type=float, default=None, help="Top-p sampling (optional)."
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty (optional).",
    )
    parser.add_argument(
        "--subtalker-top-k",
        type=int,
        default=None,
        help="Subtalker top-k (optional, only for tokenizer v2).",
    )
    parser.add_argument(
        "--subtalker-top-p",
        type=float,
        default=None,
        help="Subtalker top-p (optional, only for tokenizer v2).",
    )
    parser.add_argument(
        "--subtalker-temperature",
        type=float,
        default=None,
        help="Subtalker temperature (optional, only for tokenizer v2).",
    )

    return parser


def _collect_gen_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    mapping = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_top_k": args.subtalker_top_k,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)

        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid

    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0

        if m <= 1.0 + 1e-6:
            pass
        else:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


def _wav_to_gradio_audio(wav: np.ndarray, sr: int) -> Tuple[int, np.ndarray]:
    wav = np.asarray(wav, dtype=np.float32)
    return sr, wav


def _encode_audio_to_base64(wav: np.ndarray, sr: int) -> str:
    """Encode (wav float32 [-1,1], sr) to base64 data URL for API ref_audio."""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    wav = np.clip(wav, -1.0, 1.0)
    samples = (wav * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, samples)
    buf.seek(0)
    audio_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:audio/wav;base64,{audio_b64}"


def _decode_wav_response(content: bytes) -> Tuple[int, np.ndarray]:
    """Decode raw WAV bytes from API response to (sr, wav_float32) for Gradio."""
    buf = io.BytesIO(content)
    sr, data = wavfile.read(buf)
    if data.dtype == np.int16:
        wav = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        wav = data.astype(np.float32) / 2147483648.0
    else:
        wav = data.astype(np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    return sr, wav


def _call_tts_api(
    api_base: str,
    api_key: str,
    model: str,
    payload: Dict[str, Any],
    timeout: float = 300.0,
) -> bytes:
    """POST to /v1/audio/speech and return raw audio bytes."""
    url = f"{api_base.rstrip('/')}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {**payload, "model": model}
    if "response_format" not in payload:
        payload["response_format"] = "wav"
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type or response.content[:1] == b"{":
        try:
            data = response.json()
            message = (
                data.get("error", {}).get("message")
                if isinstance(data, dict)
                else str(data)
            )
        except Exception:
            message = response.text[:500]
        raise ValueError(f"API returned JSON instead of audio: {message}")
    return response.content


# Default options when API does not expose supported lists
DEFAULT_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "Spanish",
    "French",
    "German",
]
DEFAULT_SPEAKERS = ["Vivian", "Ryan", "Bella", "Ethan", "Emma", "Liam"]


def build_demo(
    api_base: str,
    api_key: str,
    model: str,
    gen_kwargs_default: Dict[str, Any],
) -> gr.Blocks:
    lang_choices_disp, lang_map = _build_choices_and_map(DEFAULT_LANGUAGES)
    spk_choices_disp, spk_map = _build_choices_and_map(DEFAULT_SPEAKERS)

    voice_design_presets: Dict[str, str] = {
        "Custom (write your own)": "",
        "Multi-speaker (Lucas & Mia)": (
            'Lucas: "Male, 17 years old, tenor range, gaining confidence - deeper breath support now, '
            'though vowels still tighten when nervous"\n'
            'Mia: "Female, 16 years old, mezzo-soprano range, softening - lowering register to intimate '
            'speaking voice, consonants softening"'
        ),
        "Marcus Cole (broadcast announcer)": (
            "First: Background Information\n"
            "Character Name: Marcus Cole\n"
            "Voice Profile: A bright, agile male voice with a natural upward lift, delivering lines at a brisk, "
            "energetic pace. Pitch leans high with spark, volume projects clearly - near-shouting at peaks - to convey "
            "urgency and excitement. Speech flows seamlessly, fluently, each word sharply defined, riding a current "
            "of dynamic rhythm.\n"
            "Background: Longtime broadcast booth announcer for national television, specializing in live "
            "interstitials and public engagement spots. His voice bridges segments, rallies action, and keeps "
            "momentum alive - from voter drives to entertainment news.\n"
            "Presence: Late 50s, neatly groomed, dressed in a crisp shirt under studio lights. Moves with practiced "
            "ease, eyes locked on the script, energy coiled and ready.\n"
            "Personality: Energetic, precise, inherently engaging. He doesn't just read - he propels. Behind the speed "
            'is intent: to inform fast, to move people to act. Whether it\'s "text VOTE to 5703" or a star-studded '
            "tease, he makes it feel immediate, vital."
        ),
        "Human-like (late 20s/early 30s male)": (
            "A relaxed, naturally expressive male voice in his late twenties to early thirties, with a moderately "
            "low pitch, casual speaking rate, and conversational volume; deliver lines with a light, self-deprecating "
            "tone, breaking into genuine, easygoing laughter at moments of embarrassment, while maintaining clear "
            "articulation and an overall warm, approachable clarity."
        ),
        "Gradual control (female, frustration)": (
            "gender: Female.\n"
            "pitch: Mid-range female pitch, rising sharply with frustration.\n"
            "speed: Starts measured, then accelerates rapidly during emotional outburst.\n"
            "volume: Begins conversational, escalates quickly to loud and forceful.\n"
            "age: Young adult to middle-aged.\n"
            "clarity: High clarity and distinct articulation throughout.\n"
            "fluency: Highly fluent with no significant pauses or fillers.\n"
            "accent: General American English.\n"
            "texture: Bright and clear vocal quality.\n"
            "emotion: Shifts abruptly from neutral acceptance to intense resentment and anger.\n"
            "tone: Initially accepting, becomes sharply accusatory and confrontational.\n"
            "personality: Assertive and emotionally expressive when provoked."
        ),
        "Authoritative narrator (male)": (
            "gender: Male.\n"
            "pitch: Low male pitch, generally stable.\n"
            "speed: Deliberate pace, slowing slightly after the initial exclamation.\n"
            "volume: Starts loud, then transitions to a projected conversational volume.\n"
            "age: Middle-aged adult.\n"
            "clarity: High clarity with distinct pronunciation.\n"
            "fluency: Highly fluent.\n"
            "accent: American English.\n"
            "texture: Resonant and slightly gravelly.\n"
            "emotion: Initially commanding, shifting to narrative amusement.\n"
            "tone: Authoritative start, moving to an engaging, descriptive tone.\n"
            "personality: Confident and performative."
        ),
        "Theatrical character (male)": (
            "gender: Male.\n"
            "pitch: Artificially high-pitched, slightly lowering after the initial laugh.\n"
            "speed: Rapid during the laugh, then slowing to a deliberate pace.\n"
            "volume: Loud laugh transitioning to a standard conversational level.\n"
            "age: Young adult to middle-aged, performing a character voice.\n"
            "clarity: Clear and distinct articulation.\n"
            "fluency: Fluent delivery without hesitation.\n"
            "accent: American English.\n"
            "texture: Slightly strained and somewhat nasal quality.\n"
            "emotion: Forced amusement shifting to feigned resignation.\n"
            "tone: Initially playful, then shifts to a slightly put-upon tone.\n"
            "personality: Theatrical and expressive."
        ),
        "British announcer (male)": (
            "gender: Male.\n"
            "pitch: Low male pitch with significant upward inflections for emphasis and excitement.\n"
            "speed: Fast-paced delivery with deliberate pauses for dramatic effect.\n"
            "volume: Loud and projecting, increasing notably during moments of praise and announcements.\n"
            "age: Young adult to middle-aged adult.\n"
            "clarity: Highly articulate and distinct pronunciation.\n"
            "fluency: Very fluent speech with no hesitations.\n"
            "accent: British English.\n"
            "texture: Bright and clear vocal texture.\n"
            "emotion: Enthusiastic and excited, especially when complimenting.\n"
            "tone: Upbeat, authoritative, and performative.\n"
            "personality: Confident, extroverted, and engaging."
        ),
    }

    def _gen_common_kwargs() -> Dict[str, Any]:
        return dict(gen_kwargs_default)

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = ".gradio-container {max-width: none !important;}"

    with gr.Blocks(theme=theme, css=css) as demo:
        gr.Markdown(
            f"""
# Qwen3 TTS Demo (vLLM-omni API)
**API base:** `{api_base}`  
**Model:** `{model}`  
"""
        )

        with gr.Tabs():
            with gr.Tab("CustomVoice"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_in = gr.Textbox(
                            label="Text",
                            lines=4,
                            placeholder="Enter text to synthesize.",
                        )
                        with gr.Row():
                            lang_in = gr.Dropdown(
                                label="Language",
                                choices=lang_choices_disp,
                                value="Auto",
                                interactive=True,
                            )
                            spk_in = gr.Dropdown(
                                label="Speaker",
                                choices=spk_choices_disp,
                                value="Vivian",
                                interactive=True,
                            )
                        instruct_in = gr.Textbox(
                            label="Instruction (Optional)",
                            lines=2,
                            placeholder="e.g. Say it in a very angry tone.",
                        )
                        btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=3):
                        audio_out = gr.Audio(label="Output Audio", type="numpy")
                        err = gr.Textbox(label="Status", lines=2)

                def run_instruct(
                    text: str, lang_disp: str, spk_disp: str, instruct: str
                ):
                    try:
                        if not text or not text.strip():
                            return None, "Text is required."
                        if not spk_disp:
                            return None, "Speaker is required."
                        language = lang_map.get(lang_disp, "Auto")
                        speaker = spk_map.get(spk_disp, spk_disp)
                        payload = {
                            "input": text.strip(),
                            "voice": speaker,
                            "task_type": "CustomVoice",
                            "language": language,
                            **_gen_common_kwargs(),
                        }
                        if (instruct or "").strip():
                            payload["instructions"] = (instruct or "").strip()
                        raw = _call_tts_api(api_base, api_key, model, payload)
                        sr, wav = _decode_wav_response(raw)
                        return (sr, wav), "Finished."
                    except httpx.HTTPStatusError as e:
                        return (
                            None,
                            f"API error {e.response.status_code}: {e.response.text[:500]}",
                        )
                    except Exception as e:
                        return None, f"{type(e).__name__}: {e}"

                btn.click(
                    run_instruct,
                    inputs=[text_in, lang_in, spk_in, instruct_in],
                    outputs=[audio_out, err],
                )

            with gr.Tab("VoiceDesign"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_in = gr.Textbox(
                            label="Text",
                            lines=4,
                            value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
                        )
                        with gr.Row():
                            lang_in = gr.Dropdown(
                                label="Language",
                                choices=lang_choices_disp,
                                value="Auto",
                                interactive=True,
                            )
                        preset_in = gr.Dropdown(
                            label="Preset",
                            choices=list(voice_design_presets.keys()),
                            value="Custom (write your own)",
                            interactive=True,
                        )
                        design_in = gr.Textbox(
                            label="Voice Design Instruction",
                            lines=6,
                            value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.",
                        )
                        btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=3):
                        audio_out = gr.Audio(label="Output Audio", type="numpy")
                        err = gr.Textbox(label="Status", lines=2)

                def run_voice_design(text: str, lang_disp: str, design: str):
                    try:
                        if not text or not text.strip():
                            return None, "Text is required."
                        if not design or not design.strip():
                            return (
                                None,
                                "Voice design instruction is required.",
                            )
                        language = lang_map.get(lang_disp, "Auto")
                        payload = {
                            "input": text.strip(),
                            "task_type": "VoiceDesign",
                            "instructions": design.strip(),
                            "language": language,
                            **_gen_common_kwargs(),
                        }
                        raw = _call_tts_api(api_base, api_key, model, payload)
                        sr, wav = _decode_wav_response(raw)
                        return (sr, wav), "Finished."
                    except httpx.HTTPStatusError as e:
                        return (
                            None,
                            f"API error {e.response.status_code}: {e.response.text[:500]}",
                        )
                    except Exception as e:
                        return None, f"{type(e).__name__}: {e}"

                def apply_voice_design_preset(preset_name: str):
                    preset = voice_design_presets.get(preset_name, "")
                    if preset:
                        return preset
                    return ""

                preset_in.change(
                    apply_voice_design_preset,
                    inputs=[preset_in],
                    outputs=[design_in],
                )

                btn.click(
                    run_voice_design,
                    inputs=[text_in, lang_in, design_in],
                    outputs=[audio_out, err],
                )

            with gr.Tab("Base (Voice Clone)"):
                with gr.Tabs():
                    with gr.Tab("Clone & Generate"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                ref_audio = gr.Audio(
                                    label="Reference Audio",
                                )
                                ref_text = gr.Textbox(
                                    label="Reference Text",
                                    lines=2,
                                    placeholder="Required unless x-vector only is enabled.",
                                )
                                xvec_only = gr.Checkbox(
                                    label="Use x-vector only (no reference text needed; lower quality)",
                                    value=False,
                                )

                            with gr.Column(scale=2):
                                text_in = gr.Textbox(
                                    label="Target Text",
                                    lines=4,
                                    placeholder="Enter text to synthesize.",
                                )
                                lang_in = gr.Dropdown(
                                    label="Language",
                                    choices=lang_choices_disp,
                                    value="Auto",
                                    interactive=True,
                                )
                                btn = gr.Button("Generate", variant="primary")

                            with gr.Column(scale=3):
                                audio_out = gr.Audio(label="Output Audio", type="numpy")
                                err = gr.Textbox(label="Status", lines=2)

                        def run_voice_clone(
                            ref_aud,
                            ref_txt: str,
                            use_xvec: bool,
                            text: str,
                            lang_disp: str,
                        ):
                            try:
                                if not text or not text.strip():
                                    return (
                                        None,
                                        "Target text is required.",
                                    )
                                at = _audio_to_tuple(ref_aud)
                                if at is None:
                                    return (
                                        None,
                                        "Reference audio is required.",
                                    )
                                if (not use_xvec) and (
                                    not ref_txt or not ref_txt.strip()
                                ):
                                    return None, (
                                        "Reference text is required when x-vector only is not enabled."
                                    )
                                language = lang_map.get(lang_disp, "Auto")
                                wav, sr = at
                                ref_audio_b64 = _encode_audio_to_base64(wav, sr)
                                payload = {
                                    "input": text.strip(),
                                    "task_type": "Base",
                                    "ref_audio": ref_audio_b64,
                                    "ref_text": (ref_txt.strip() if ref_txt else None)
                                    or "",
                                    "x_vector_only_mode": bool(use_xvec),
                                    "language": language,
                                    **_gen_common_kwargs(),
                                }
                                raw = _call_tts_api(api_base, api_key, model, payload)
                                out_sr, out_wav = _decode_wav_response(raw)
                                return (out_sr, out_wav), "Finished."
                            except httpx.HTTPStatusError as e:
                                return (
                                    None,
                                    f"API error {e.response.status_code}: {e.response.text[:500]}",
                                )
                            except Exception as e:
                                return None, f"{type(e).__name__}: {e}"

                        btn.click(
                            run_voice_clone,
                            inputs=[ref_audio, ref_text, xvec_only, text_in, lang_in],
                            outputs=[audio_out, err],
                        )

                    with gr.Tab("Save / Load Voice"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.Markdown(
                                    """
### Save Voice
Upload reference audio and text, choose x-vector only or not, then save a reusable voice file (zip with ref.wav + meta.json).
"""
                                )
                                ref_audio_s = gr.Audio(
                                    label="Reference Audio", type="numpy"
                                )
                                ref_text_s = gr.Textbox(
                                    label="Reference Text",
                                    lines=2,
                                    placeholder="Required unless x-vector only is enabled.",
                                )
                                xvec_only_s = gr.Checkbox(
                                    label="Use x-vector only (no reference text needed; lower quality)",
                                    value=False,
                                )
                                save_btn = gr.Button(
                                    "Save Voice File", variant="primary"
                                )
                                prompt_file_out = gr.File(label="Voice File")

                            with gr.Column(scale=2):
                                gr.Markdown(
                                    """
### Load Voice & Generate
Upload a previously saved voice file (.zip), then synthesize new text.
"""
                                )
                                prompt_file_in = gr.File(label="Upload Voice File")
                                text_in2 = gr.Textbox(
                                    label="Target Text",
                                    lines=4,
                                    placeholder="Enter text to synthesize.",
                                )
                                lang_in2 = gr.Dropdown(
                                    label="Language",
                                    choices=lang_choices_disp,
                                    value="Auto",
                                    interactive=True,
                                )
                                gen_btn2 = gr.Button("Generate", variant="primary")

                            with gr.Column(scale=3):
                                audio_out2 = gr.Audio(
                                    label="Output Audio", type="numpy"
                                )
                                err2 = gr.Textbox(label="Status", lines=2)

                        def save_prompt(ref_aud, ref_txt: str, use_xvec: bool):
                            try:
                                at = _audio_to_tuple(ref_aud)
                                if at is None:
                                    return (
                                        None,
                                        "Reference audio is required.",
                                    )
                                if (not use_xvec) and (
                                    not ref_txt or not ref_txt.strip()
                                ):
                                    return None, (
                                        "Reference text is required when x-vector only is not enabled."
                                    )
                                wav, sr = at
                                fd, out_path = tempfile.mkstemp(
                                    prefix="voice_clone_", suffix=".zip"
                                )
                                os.close(fd)
                                with zipfile.ZipFile(
                                    out_path, "w", zipfile.ZIP_DEFLATED
                                ) as zf:
                                    buf = io.BytesIO()
                                    samples = (
                                        np.clip(wav, -1.0, 1.0) * 32767.0
                                    ).astype(np.int16)
                                    wavfile.write(buf, sr, samples)
                                    buf.seek(0)
                                    zf.writestr("ref.wav", buf.read())
                                    zf.writestr(
                                        "meta.json",
                                        json.dumps(
                                            {
                                                "ref_text": (ref_txt or "").strip()
                                                or None,
                                                "x_vector_only_mode": bool(use_xvec),
                                            },
                                            indent=2,
                                        ),
                                    )
                                return out_path, "Finished."
                            except Exception as e:
                                return None, f"{type(e).__name__}: {e}"

                        def load_prompt_and_gen(file_obj, text: str, lang_disp: str):
                            try:
                                if file_obj is None:
                                    return (
                                        None,
                                        "Voice file is required.",
                                    )
                                if not text or not text.strip():
                                    return (
                                        None,
                                        "Target text is required.",
                                    )
                                path = (
                                    getattr(file_obj, "name", None)
                                    or getattr(file_obj, "path", None)
                                    or str(file_obj)
                                )
                                with zipfile.ZipFile(path, "r") as zf:
                                    if (
                                        "ref.wav" not in zf.namelist()
                                        or "meta.json" not in zf.namelist()
                                    ):
                                        return (
                                            None,
                                            "Invalid voice file: missing ref.wav or meta.json.",
                                        )
                                    wav_bytes = zf.read("ref.wav")
                                    meta = json.loads(
                                        zf.read("meta.json").decode("utf-8")
                                    )
                                ref_audio_b64 = base64.b64encode(wav_bytes).decode(
                                    "utf-8"
                                )
                                ref_audio_b64 = f"data:audio/wav;base64,{ref_audio_b64}"
                                ref_text = meta.get("ref_text") or ""
                                x_vector_only = meta.get("x_vector_only_mode", False)
                                language = lang_map.get(lang_disp, "Auto")
                                payload = {
                                    "input": text.strip(),
                                    "task_type": "Base",
                                    "ref_audio": ref_audio_b64,
                                    "ref_text": ref_text,
                                    "x_vector_only_mode": x_vector_only,
                                    "language": language,
                                    **_gen_common_kwargs(),
                                }
                                raw = _call_tts_api(api_base, api_key, model, payload)
                                out_sr, out_wav = _decode_wav_response(raw)
                                return (out_sr, out_wav), "Finished."
                            except httpx.HTTPStatusError as e:
                                return (
                                    None,
                                    f"API error {e.response.status_code}: {e.response.text[:500]}",
                                )
                            except Exception as e:
                                return None, (
                                    f"Failed to read or use voice file. Check file format/content.\n"
                                    f"{type(e).__name__}: {e}"
                                )

                        save_btn.click(
                            save_prompt,
                            inputs=[ref_audio_s, ref_text_s, xvec_only_s],
                            outputs=[prompt_file_out, err2],
                        )
                        gen_btn2.click(
                            load_prompt_and_gen,
                            inputs=[prompt_file_in, text_in2, lang_in2],
                            outputs=[audio_out2, err2],
                        )

        gr.Markdown(
            """
**Disclaimer**  
The audio is automatically generated/synthesized by an AI model solely to demonstrate the model's capabilities; it may be inaccurate or inappropriate, does not represent the views of the developer/operator, and does not constitute professional advice. You are solely responsible for evaluating, using, distributing, or relying on this audio; to the maximum extent permitted by applicable law, the developer/operator disclaims liability for any direct, indirect, incidental, or consequential damages arising from the use of or inability to use the audio, except where liability cannot be excluded by law. Do not use this service to intentionally generate or replicate unlawful, harmful, defamatory, fraudulent, deepfake, or privacy/publicity/copyright/trademark-infringing content; if a user prompts, supplies materials, or otherwise facilitates any illegal or infringing conduct, the user bears all legal consequences and the developer/operator is not responsible.
"""
        )

    return demo


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    gen_kwargs_default = _collect_gen_kwargs(args)
    demo = build_demo(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        gen_kwargs_default=gen_kwargs_default,
    )

    launch_kwargs: Dict[str, Any] = dict(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        ssl_verify=True if args.ssl_verify else False,
    )
    if args.ssl_certfile is not None:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile is not None:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile

    demo.queue(default_concurrency_limit=int(args.concurrency)).launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
