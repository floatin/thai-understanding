"""
Demo: Run pipeline on actual Thai-SUP samples.
This is a smoke test - the adapter is untrained so outputs are not meaningful.
But it proves the pipeline can decode Thai audio + task prompts.
"""
import warnings
warnings.filterwarnings("ignore")
import io
import time
import torch
import soundfile as sf
import pyarrow.parquet as pq
from pathlib import Path

import sys
sys.path.insert(0, '/data/workspace/asr-model-training/thai-understanding/src')
from slu_pipeline import ThaiSLUPipeline


def load_sample(parquet_path: str, idx: int = 0):
    """Load a single sample from a Thai-SUP parquet shard."""
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    row = df.iloc[idx]
    audio_bytes = row['audio_flac']
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    return {
        'task_id': row['task_id'],
        'task_prompt': row['task_prompt'],
        'text': row['text'],
        'label': row['label'],
        'audio': torch.from_numpy(audio).unsqueeze(0),  # (1, T)
        'sr': sr,
        'duration_s': row['duration_s'],
    }


def main():
    base = Path("/data/workspace/asr-model-training/thai-understanding")
    print("Loading pipeline...")
    pipeline = ThaiSLUPipeline(
        encoder_ckpt=str(base / "XLSR-Thai/checkpoint_best.pt"),
        llm_path=str(base / "Typhoon2-3B"),
        encoder_device="cuda",
        llm_device="cpu",
    )
    print()

    # Run on one sample from each task
    for task, parquet in [
        ("IC", base / "Thai-SUP/IC/dev/dev-00000.parquet"),
        ("NER", base / "Thai-SUP/NER/dev/dev-00000.parquet"),
        ("SR", base / "Thai-SUP/SR/dev/dev-00000.parquet"),
    ]:
        sample = load_sample(str(parquet), idx=0)
        print(f"\n=== Task: {sample['task_id']} ===")
        print(f"  Audio: {sample['duration_s']:.1f}s @ {sample['sr']} Hz")
        print(f"  Text: {sample['text']!r}")
        print(f"  Label: {sample['label']!r}")
        print(f"  Prompt (first 80 chars): {sample['task_prompt'][:80]!r}...")

        t0 = time.time()
        out = pipeline.generate(sample['audio'], sample['task_prompt'], max_new_tokens=24)
        dt = time.time() - t0
        print(f"  Generated ({dt:.1f}s): {out!r}")
        print(f"  NOTE: Adapter is untrained — output is nonsensical.")


if __name__ == "__main__":
    main()
