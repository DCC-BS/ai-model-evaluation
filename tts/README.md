# TTS demo harness

This is a small demo harness for text-to-speech (TTS). It starts a Qwen3-TTS model on vLLM-Omni and puts a Gradio page in front of it. We use it to generate speech from test sentences and to judge the result by listening, following [`LISTENING_PROTOCOL.md`](LISTENING_PROTOCOL.md).

It is not an automated benchmark. There is no metric and no score file. We do not know of an automated measure that tells us whether a German or Swiss voice pronounces a Basel place name acceptably, so people listen.

## Modes

The demo supports the three modes of Qwen3-TTS. See the [Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS) and the [announcement](https://qwen.ai/blog?id=qwen3tts-0115) for details.

- **CustomVoice.** Pick a built-in voice and give the text. An optional instruction can steer the tone.
- **VoiceDesign.** Describe a voice in plain language (age, mood, style) and the model generates one.
- **Base.** Give a short reference recording and the model speaks new text in that voice. Only use recordings of people who agreed to it.

## Install

The demo needs a vLLM build that matches your Torch backend. `--torch-backend=auto` is an installer option that `pyproject.toml` cannot express, so it is a second step after `uv sync`:

```bash
cd tts
uv sync
uv pip install vllm==0.14.0 --torch-backend=auto
```

Do not run a plain `uv sync` or `uv run` afterwards: either one resets vLLM to the default build. `run_server.sh` calls `uv run --no-sync` for this reason. Run your own commands the same way.

## How to run

`run_server.sh` starts the vLLM-Omni server and the Gradio UI together:

```bash
./run_server.sh                 # VoiceDesign (default)
./run_server.sh CustomVoice
./run_server.sh Base
API_PORT=9000 GRADIO_PORT=8860 ./run_server.sh   # other ports
```

The API listens on port 8000 and the UI on port 7860 by default. The API binds to 127.0.0.1. To expose the API on another interface, set `HOST`, for example `HOST=0.0.0.0 ./run_server.sh`. The script stops the server when you exit.

## Limits

- Judged by listening only. Ratings depend on who listens, so the protocol asks for at least two listeners.
- One model family. Other TTS models need their own server setup.
- The scripts are tied to the vLLM and vLLM-Omni versions in `pyproject.toml`.

## Layout

```
main.py                # Gradio demo client for the /v1/audio/speech API
run_server.sh          # start server and UI
qwen3_tts.yaml         # vLLM-Omni stage config
LISTENING_PROTOCOL.md  # how we rate what we hear
pyproject.toml
```
