"""Fetch a small public sample (FLEURS, German) in the layout the suite expects.

FLEURS (google/fleurs, CC-BY-4.0) is read speech in Standard German with no
Swiss German. The sample shows that the pipeline runs end to end, nothing more.

    uv run --extra sample python scripts/fetch_sample.py --out data/sample --n 10
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import soundfile as sf


def write_case(
    out_dir: Path,
    case_id: str,
    audio_array,
    sampling_rate: int,
    transcript: str,
) -> Path:
    """Write one test case: RecordedAudio.wav + ground_truth_transcript.txt."""
    case_dir = Path(out_dir) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    sf.write(case_dir / "RecordedAudio.wav", audio_array, sampling_rate)
    (case_dir / "ground_truth_transcript.txt").write_text(transcript, encoding="utf-8")
    return case_dir


def fetch(out_dir: Path, n: int) -> list[Path]:
    from datasets import Audio, load_dataset

    ds = load_dataset("google/fleurs", "de_de", split="test", streaming=True)
    # Keep raw bytes and decode with soundfile, so torchcodec is not required.
    ds = ds.cast_column("audio", Audio(decode=False))
    cases: list[Path] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        array, rate = sf.read(io.BytesIO(row["audio"]["bytes"]))
        cases.append(
            write_case(
                out_dir,
                f"case_{i + 1:03d}",
                array,
                rate,
                row["transcription"],
            )
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/sample"))
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    cases = fetch(args.out, args.n)
    print(f"Wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
