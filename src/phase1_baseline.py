"""
Phase 1: Baseline (ZikXewen XLSR-53-Thai) error analysis.
Stratified by duration and text length to identify hard cases.
"""
import warnings
warnings.filterwarnings("ignore")
import io, json, time, random
import numpy as np
import torch
import soundfile as sf
import pyarrow.parquet as pq
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor


def cer_ns(ref, hyp):
    n = max(len(ref), 1)
    m = len(hyp)
    if m == 0:
        return 1.0
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


def main():
    base = Path("/data/workspace/asr-model-training/thai-understanding")
    print("[1/3] Loading ZikXewen XLSR-53-Thai...")
    processor = Wav2Vec2Processor(
        feature_extractor=Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False),
        tokenizer=Wav2Vec2CTCTokenizer(
            vocab_file=str(base / "ZikXewen-thai-ctc/vocab.json"),
            unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|"),
    )
    model = Wav2Vec2ForCTC.from_pretrained(str(base / "ZikXewen-thai-ctc")).cuda().eval()

    print(f"\n[2/3] Running on 2000 dev samples (per task)...")
    all_results = []
    for task in ["IC", "NER", "SR"]:
        for subset in ["dev", "test"]:
            df = pq.read_table(str(base / f"Thai-SUP/{task}/{subset}/{subset}-00000.parquet")).to_pandas()
            cers = []
            t0 = time.time()
            with torch.no_grad():
                for _, row in df.iterrows():
                    audio, sr = sf.read(io.BytesIO(row['audio_flac']), dtype='float32')
                    if len(audio) < 3200:
                        audio = np.pad(audio, (0, 3200 - len(audio)))
                    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.cuda()
                    pred = processor.batch_decode(torch.argmax(model(inputs).logits, dim=-1))[0]
                    ref = str(row['text'])
                    cers.append({
                        'cer': cer_ns(ref, pred),
                        'task': task,
                        'subset': subset,
                        'duration': float(row['duration_s']),
                        'text_len': len(ref),
                        'pred': pred,
                        'ref': ref,
                    })
            avg = np.mean([r['cer'] for r in cers])
            print(f"  {task}/{subset}: avg CER={avg:.2%}  ({time.time()-t0:.0f}s)")
            all_results.extend(cers)

    # Stratified analysis
    print(f"\n[3/3] Stratified error analysis (6000 total samples)...")
    by_duration = {'<3s': [], '3-6s': [], '6-10s': [], '>10s': []}
    by_textlen = {'<30': [], '30-60': [], '60-100': [], '>100': []}
    by_task = {'IC': [], 'NER': [], 'SR': []}
    for r in all_results:
        d = r['duration']
        if d < 3: by_duration['<3s'].append(r['cer'])
        elif d < 6: by_duration['3-6s'].append(r['cer'])
        elif d < 10: by_duration['6-10s'].append(r['cer'])
        else: by_duration['>10s'].append(r['cer'])
        t = r['text_len']
        if t < 30: by_textlen['<30'].append(r['cer'])
        elif t < 60: by_textlen['30-60'].append(r['cer'])
        elif t < 100: by_textlen['60-100'].append(r['cer'])
        else: by_textlen['>100'].append(r['cer'])
        by_task[r['task']].append(r['cer'])

    print(f"\n按 duration 分层:")
    for k, v in by_duration.items():
        if v: print(f"  {k:<8} n={len(v):>5} avg CER={np.mean(v):.2%}")
    print(f"\n按 text_len 分层:")
    for k, v in by_textlen.items():
        if v: print(f"  {k:<8} n={len(v):>5} avg CER={np.mean(v):.2%}")
    print(f"\n按 task 分层:")
    for k, v in by_task.items():
        if v: print(f"  {k:<8} n={len(v):>5} avg CER={np.mean(v):.2%}")

    # Worst samples (where to focus Stage 2 fine-tuning)
    all_results.sort(key=lambda r: -r['cer'])
    print(f"\n=== CER 最差的 5 条 (优先改进) ===")
    for r in all_results[:5]:
        print(f"  [{r['task']}/{r['subset']}] dur={r['duration']:.1f}s len={r['text_len']} CER={r['cer']:.2%}")
        print(f"    ref: {r['ref'][:80]!r}")
        print(f"    hyp: {r['pred'][:80]!r}")

    # Save report
    report = {
        'model': 'ZikXewen XLSR-53-Thai (CommonVoice fine-tuned)',
        'total_samples': len(all_results),
        'avg_cer_no_space': float(np.mean([r['cer'] for r in all_results])),
        'by_duration': {k: {'n': len(v), 'avg_cer': float(np.mean(v))} for k, v in by_duration.items()},
        'by_text_len': {k: {'n': len(v), 'avg_cer': float(np.mean(v))} for k, v in by_textlen.items()},
        'by_task': {k: {'n': len(v), 'avg_cer': float(np.mean(v))} for k, v in by_task.items()},
        'worst_samples': all_results[:10],
    }
    out = base / "phase1_baseline_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
