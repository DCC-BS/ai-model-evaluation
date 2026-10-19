# Data policy

This repository contains our test method and code. It contains no internal test data, no results and no model outputs.

## What data we use

- **Public datasets.** The code downloads them when it runs, and we never store them in the repository. Examples are FLEURS for speech and `fhswf/german_handwriting` for handwriting.
- **Made-up samples**, where no suitable public data exists.
- **Our own data**, which stays private. It includes recordings of real work conversations, internal documents and internal tasks.

## Why our data stays private

Our recordings and documents contain personal data and the work of people who have not agreed to have it published. Transcripts and model outputs made from them contain the same information. So we publish neither the data nor the results.

This means you can't reproduce our internal numbers with this repository. You can run the same tests on your own data.

## Using your own data

### STT

Create one folder per test case in `data/`, each with a `RecordedAudio.wav` and a `ground_truth_transcript.txt`. Then run `asr-eval` with `--data-dir`. See [`stt/README.md`](../stt/README.md).

### OCR

The test loads a public handwriting dataset. To use your own images, change the part of `ocr/eval_ocr.py` that loads the dataset. You need one correct text per image. See [`ocr/README.md`](../ocr/README.md).

### TTS

Replace the test sentences in [`tts/LISTENING_PROTOCOL.md`](../tts/LISTENING_PROTOCOL.md) with sentences from your own work. Only use text you are allowed to use.

### PII/NER and decision models

This code isn't public yet. When we add it, it will follow the same rule: public or made-up samples in the repository, your own data stays with you.

## Rules for contributors

- Don't commit recordings, documents, transcripts, model outputs or result files.
- Don't commit names of people from internal recordings.
- Don't commit server names, internal URLs, keys or `.env` files.
- Run the safety scan before you commit. The README explains how.
