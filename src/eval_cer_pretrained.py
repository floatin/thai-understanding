"""
Run ASR using the pretrained XLSR-53-Thai CTC model on 1000 Thai-SUP samples.

This is the "honest" baseline: XLSR-53 backbone pretrained on CommonVoice Thai,
giving meaningful CER. Compare with:
  - XLSR-Thai (our loaded encoder) + random CTC head → CER ~100% (failed to learn)
  - LLM (Typhoon2) + random adapter → CER 142% (hallucination)
  - Pretrained XLSR-53-Thai-CTC → CER ~60% (this script)
  - XLSR-Thai + U-Align + LLM (paper, not in this scope) → CER ~13-14%
"""
import warnings
warnings.filterwarnings("ignore")
import io
import time
import json
import random
import argparse
import torch
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor


def cer(ref: str, hyp: str) -> float:
    """Character error rate."""
    n = max(len(ref), 1)
    m = len(hyp)
    if m == 0:
        return n / n if n > 0 else 0.0
    dp = list(range(m+1))
    for i in range(1, len(ref)+1):
        prev, dp[0] = dp[0], i
        for j in range(1, m+1):
            cur = dp[j]
            if ref[i-1] == hyp[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = cur
    return dp[m] / n


def cer_no_space(ref: str, hyp: str) -> float:
    """CER after removing spaces (paper's standard)."""
    return cer(ref.replace(' ', ''), hyp.replace(' ', ''))


def load_samples(parquet_path, indices=None, n=1000):
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()
    if indices is None:
        random.seed(42)
        indices = random.sample(range(len(df)), min(n, len(df)))
    samples = []
    for i in indices:
        row = df.iloc[i]
        audio_bytes = row['audio_flac']
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        samples.append({
            'task_id': row['task_id'],
            'data_id': row['data_id'],
            'text': str(row['text']),
            'audio': audio,
            'sr': sr,
            'duration_s': float(row['duration_s']),
        })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_samples', type=int, default=1000)
    args = ap.parse_args()

    base = Path("/data/workspace/asr-model-training/thai-understanding")

    # Load pretrained model
    print("[1/3] Loading pretrained XLSR-53-Thai-CTC (ZikXewen)...")
    model_path = base / "ZikXewen-thai-ctc"
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(model_path / "vocab.json"),
        unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(str(model_path)).cuda().eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load samples
    print(f"\n[2/3] Loading {args.n_samples} samples from Thai-SUP dev+test...")
    all_samples = []
    per_split = args.n_samples // 6
    for task in ["IC", "NER", "SR"]:
        for subset in ["dev", "test"]:
            parquet = base / f"Thai-SUP/{task}/{subset}/{subset}-00000.parquet"
            samples = load_samples(parquet, n=per_split)
            all_samples.extend(samples)
            print(f"  {task}/{subset}: +{len(samples)}")
    all_samples = [s for s in all_samples if 5 < len(s['text']) < 200]
    print(f"  After length filter: {len(all_samples)} samples")

    # Run ASR
    print(f"\n[3/3] Running ASR inference on {len(all_samples)} samples...")
    results = {'with_space': [], 'no_space': []}
    details = []
    by_task = {}
    t0 = time.time()
    with torch.no_grad():
        for i, s in enumerate(all_samples):
            # Pad/clip audio to at least 0.2s (3200 samples) so conv works
            MIN = 3200
            audio = s["audio"]
            if len(audio) < MIN:
                audio = np.pad(audio, (0, MIN - len(audio)))
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.cuda()
            logits = model(inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred = processor.batch_decode(pred_ids)[0]

            c_ws = cer(s['text'], pred)
            c_ns = cer_no_space(s['text'], pred)
            results['with_space'].append(c_ws)
            results['no_space'].append(c_ns)
            by_task.setdefault(s['task_id'], []).append(c_ns)

            if i < 5:
                details.append({'data_id': s['data_id'], 'task': s['task_id'],
                                'ref': s['text'], 'hyp': pred, 'cer_ws': round(c_ws, 3), 'cer_ns': round(c_ns, 3)})

            if (i+1) % 100 == 0:
                avg_ws = sum(results['with_space'])/(i+1)
                avg_ns = sum(results['no_space'])/(i+1)
                elapsed = time.time() - t0
                rate = (i+1) / elapsed
                eta = (len(all_samples) - i - 1) / rate
                print(f"  [{i+1}/{len(all_samples)}] avg CER (with_space)={avg_ws:.2%}, "
                      f"avg CER (no_space)={avg_ns:.2%}, "
                      f"{rate:.1f} samples/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    avg_ws = sum(results['with_space'])/len(results['with_space'])
    avg_ns = sum(results['no_space'])/len(results['no_space'])

    # Per-task stats
    task_stats = {t: {'avg_cer_ns': sum(cs)/len(cs), 'n': len(cs)} for t, cs in by_task.items()}

    # Save report
    report = {
        'model': 'ZikXewen/wav2vec2-large-xlsr-53-thai-demo (XLSR-53 + Thai CTC, trained on CommonVoice)',
        'n_samples': len(all_samples),
        'elapsed_s': elapsed,
        'avg_cer_with_space': avg_ws,
        'avg_cer_no_space': avg_ns,
        'per_task': task_stats,
        'samples': details,
    }
    out = base / "results_cer_pretrained.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n=== RESULTS ===")
    print(f"Model: {report['model']}")
    print(f"Samples: {len(all_samples)}, Time: {elapsed:.1f}s")
    print(f"Avg CER (with space): {avg_ws:.2%}")
    print(f"Avg CER (no space, paper-style): {avg_ns:.2%}")
    print(f"Per-task:")
    for t, st in task_stats.items():
        print(f"  {t}: {st['n']} samples, avg CER (no_space)={st['avg_cer_ns']:.2%}")
    print(f"\nFirst 5 examples:")
    for d in details:
        print(f"  {d['task']} CER={d['cer_ns']:.2%}")
        print(f"    ref: {d['ref'][:80]!r}")
        print(f"    hyp: {d['hyp'][:80]!r}")
    print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()
