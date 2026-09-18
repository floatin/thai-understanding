"""
Run ASR using XLSR-Thai encoder + CTC head transferred from pretrained XLSR-53-Thai.

This tests: how does the released XLSR-Thai encoder (pretrained on 36k hours Thai)
compare to the original XLSR-53 when used with the same Thai CTC head?

Method:
  1. Load pretrained XLSR-53-Thai (ZikXewen) - has CTC head trained on CommonVoice Thai
  2. Load XLSR-Thai (mcshao) - encoder only
  3. The CTC head weights are tightly coupled to the encoder, so we can't directly
     swap encoders. But we can compare:
     a) XLSR-53-Thai: encoder + head (same backbone, CommonVoice Thai fine-tuned)
     b) XLSR-Thai: just encoder, same CTC head architecture but untrained → expect high CER
     c) XLSR-Thai features → CTC head from XLSR-53-Thai: should be meaningful since
        XLSR-Thai was pretrained on more Thai data
"""
import warnings
warnings.filterwarnings("ignore")
import io
import time
import json
import random
import argparse
import numpy as np
import torch
import soundfile as sf
import pyarrow.parquet as pq
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

import sys
sys.path.insert(0, '/data/workspace/asr-model-training/thai-understanding/src')
from xlsr_thai import load_xlsr_thai, extract_features


def cer(ref, hyp):
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


def cer_ns(ref, hyp):
    return cer(ref.replace(' ', ''), hyp.replace(' ', ''))


def load_samples(parquet_path, n=1000):
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()
    random.seed(42)
    indices = random.sample(range(len(df)), min(n, len(df)))
    samples = []
    for i in indices:
        row = df.iloc[i]
        audio, sr = sf.read(io.BytesIO(row['audio_flac']), dtype='float32')
        samples.append({
            'task_id': row['task_id'],
            'data_id': row['data_id'],
            'text': str(row['text']),
            'audio': audio,
            'sr': sr,
        })
    return samples


def main():
    base = Path("/data/workspace/asr-model-training/thai-understanding")

    # Load samples
    print("[1/3] Loading samples...")
    all_samples = []
    for task in ["IC", "NER", "SR"]:
        for subset in ["dev", "test"]:
            samples = load_samples(base / f"Thai-SUP/{task}/{subset}/{subset}-00000.parquet", n=165)
            all_samples.extend(samples)
    all_samples = [s for s in all_samples if 5 < len(s['text']) < 200][:990]
    print(f"  Total: {len(all_samples)} samples")

    # Method A: Pretrained XLSR-53-Thai (ZikXewen) baseline
    print(f"\n[2/3] Loading pretrained XLSR-53-Thai (ZikXewen)...")
    model_path = base / "ZikXewen-thai-ctc"
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(model_path / "vocab.json"),
        unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    pretrained = Wav2Vec2ForCTC.from_pretrained(str(model_path)).cuda().eval()
    print(f"  Params: {sum(p.numel() for p in pretrained.parameters())/1e6:.1f}M")

    cers_pretrained = []
    t0 = time.time()
    print(f"  Running inference on {len(all_samples)} samples...")
    with torch.no_grad():
        for i, s in enumerate(all_samples):
            audio = s['audio']
            if len(audio) < 3200:
                audio = np.pad(audio, (0, 3200 - len(audio)))
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.cuda()
            logits = pretrained(inputs).logits
            pred = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
            cers_pretrained.append(cer_ns(s['text'], pred))
            if (i+1) % 200 == 0:
                avg = sum(cers_pretrained)/(i+1)
                elapsed = time.time() - t0
                print(f"    [{i+1}/{len(all_samples)}] avg CER={avg:.2%}, {elapsed:.0f}s")
    avg_pre = sum(cers_pretrained)/len(cers_pretrained)
    print(f"  XLSR-53-Thai (pretrained): CER={avg_pre:.2%}  ({time.time()-t0:.1f}s)")

    # Method B: XLSR-Thai (mcshao) encoder + transferred CTC head weights from pretrained
    # The CTC head is the lm_head of Wav2Vec2ForCTC. Its weight matrix is (vocab, hidden).
    # We can extract the encoder features from XLSR-Thai and apply the pretrained lm_head.
    # The architectures are compatible (both XLSR-53 based).
    print(f"\n[3/3] Loading XLSR-Thai (mcshao) and trying transferred CTC head...")
    xlsr_thai, cfg = load_xlsr_thai(str(base / "XLSR-Thai/checkpoint_best.pt"), device='cuda')

    # Extract CTC head weights from pretrained model
    pretrained_lm_head_weight = pretrained.lm_head.weight.detach()  # (71, 1024)
    pretrained_lm_head_bias = pretrained.lm_head.bias.detach()  # (71,)
    print(f"  CTC head weight: {pretrained_lm_head_weight.shape}, vocab={pretrained.config.vocab_size}")

    cers_xlsr_thai = []
    t0 = time.time()
    with torch.no_grad():
        for i, s in enumerate(all_samples):
            audio = s['audio']
            if len(audio) < 3200:
                audio = np.pad(audio, (0, 3200 - len(audio)))
            audio_t = torch.from_numpy(audio).unsqueeze(0).cuda()
            length = torch.tensor([audio_t.shape[1]]).cuda()
            # XLSR-Thai features
            feats, _ = extract_features(xlsr_thai, audio_t, length)  # (1, T', 1024)
            # Apply pretrained CTC head
            logits = torch.nn.functional.linear(feats, pretrained_lm_head_weight, pretrained_lm_head_bias)
            pred = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
            cers_xlsr_thai.append(cer_ns(s['text'], pred))
            if (i+1) % 200 == 0:
                avg = sum(cers_xlsr_thai)/(i+1)
                elapsed = time.time() - t0
                print(f"    [{i+1}/{len(all_samples)}] avg CER={avg:.2%}, {elapsed:.0f}s")
    avg_xt = sum(cers_xlsr_thai)/len(cers_xlsr_thai)
    print(f"  XLSR-Thai + transferred head: CER={avg_xt:.2%}  ({time.time()-t0:.1f}s)")

    print(f"\n=== COMPARISON ===")
    print(f"  XLSR-53-Thai (CommonVoice-finetuned): CER = {avg_pre:.2%}")
    print(f"  XLSR-Thai (mcshao SSL) + transferred CTC head: CER = {avg_xt:.2%}")
    print(f"  XLSR-Thai paper Table 1 (Giga2 Test, full ASR fine-tune): 13.91%")

    report = {
        'n_samples': len(all_samples),
        'pretrained_xlsr_53_thai_cer': avg_pre,
        'xlsr_thai_mcshao_transferred_head_cer': avg_xt,
        'paper_table1_giga2_xlsr_thai_ctc_cer': 0.1391,
    }
    out = base / "results_xlsr_thai_vs_pretrained.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()
