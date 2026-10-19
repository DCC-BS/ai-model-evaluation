# Data policy

This repository contains method and code. It contains no test data from internal sources, no results and no model predictions.

## What we use

- **Public datasets**, fetched at runtime and never stored in the repository. Examples are FLEURS for speech and `fhswf/german_handwriting` for handwriting.
- **Synthetic samples**, where no suitable public data exists.
- **Our own data**, kept private. It includes recordings of real work conversations, internal documents and internal tasks.

## Why our data stays private

Our recordings and documents contain personal data and work content of people who have not agreed to publication. Transcripts and predictions derived from them carry the same content. We therefore publish neither the data nor results computed on it.

The consequence is a limit we accept: you cannot reproduce our internal numbers from this repository. You can reproduce the method on your own data.

## Plug in your own data

### STT

Create one subdirectory per case in `data/`, with `RecordedAudio.wav` and `ground_truth_transcript.txt`. Run `asr-eval` with `--data-dir`. See [`stt/README.md`](../stt/README.md).

### OCR

The suite loads a public handwriting dataset. To use your own images, change the dataset loading in `ocr/eval_ocr.py`. Keep one ground-truth text per image. See [`ocr/README.md`](../ocr/README.md).

### TTS

Replace the test sentences in [`tts/LISTENING_PROTOCOL.md`](../tts/LISTENING_PROTOCOL.md) with sentences from your own domain. Use text you have the right to use.

### PII/NER and decision models

The code is not public yet. When it joins, it will follow the same rule: public or synthetic samples in the repository, your own data on your side.

## Rules for contributors

- Do not commit recordings, documents, transcripts, predictions or result files.
- Do not commit names of people from internal recordings.
- Do not commit hostnames, internal URLs, keys or `.env` files.
- Run the safety scan before you commit. See the README.
