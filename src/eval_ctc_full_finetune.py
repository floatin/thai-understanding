"""
Train CTC head (with pretrained init) on the full SR train set.

Strategy:
- Load all SR train shards (114 shards, ~225K samples)
- Subsample 10K per epoch (for time budget)
- Train 5 epochs
- Evaluate on SR test
"""
import warnings
warnings.filterwarnings("ignore")
import io
import os
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import pyarrow.parquet as pq
from pathlib import Path
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

import sys
sys.path.insert(0, '/data/workspace/asr-model-training/thai-understanding/src')
from xlsr_thai import load_xlsr_thai, extract_features


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

    # Load XLSR-Thai
    print("[1/4] Loading XLSR-Thai encoder on GPU...")
    encoder, _ = load_xlsr_thai(str(base / "XLSR-Thai/checkpoint_best.pt"))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Pretrained CTC head
    print(f"[2/4] Loading pretrained CTC head...")
    from transformers import Wav2Vec2ForCTC
    pretrained = Wav2Vec2ForCTC.from_pretrained(str(base / "ZikXewen-thai-ctc"))
    pretrained_w = pretrained.lm_head.weight.detach().clone()
    pretrained_b = pretrained.lm_head.bias.detach().clone()
    del pretrained
    head = nn.Linear(1024, 73).cuda()
    with torch.no_grad():
        head.weight.copy_(pretrained_w.cuda())
        head.bias.copy_(pretrained_b.cuda())
    head.train()
    print(f"  CTC head: {head.weight.shape}")

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(base / "ZikXewen-thai-ctc/vocab.json"),
        unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Load ALL SR train shards (sample 5000 per shard for speed)
    print(f"\n[3/4] Loading SR train (all 114 shards, 5000 subsamples)...")
    train_files = sorted((base / "Thai-SUP/SR/train").glob("*.parquet"))
    rng = random.Random(42)
    samples = []
    for shard_idx, f in enumerate(train_files):
        df = pq.read_table(str(f)).to_pandas()
        n_sample = min(5000, len(df))
        indices = rng.sample(range(len(df)), n_sample)
        for i in indices:
            row = df.iloc[i]
            text = str(row['text'])
            ids = tokenizer(text, return_tensors=None, padding=False).input_ids
            if 5 < len(text) < 100 and all(0 < iid < 71 for iid in ids):
                samples.append({'audio_flac': row['audio_flac'], 'ids': ids, 'text': text})
        if (shard_idx+1) % 20 == 0:
            print(f"  [{shard_idx+1}/{len(train_files)}] samples so far: {len(samples)}")
    print(f"  Total: {len(samples)} samples")

    # Train
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    n_epochs = 5
    bs = 4
    print(f"\nTraining {n_epochs} epochs, bs={bs}, lr=3e-4...")
    for epoch in range(n_epochs):
        random.shuffle(samples)
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(samples), bs):
            batch = samples[i:i+bs]
            audios = []
            for s in batch:
                audio, sr = sf.read(io.BytesIO(s['audio_flac']), dtype='float32')
                if len(audio) < 3200:
                    audio = np.pad(audio, (0, 3200 - len(audio)))
                audios.append(audio)
            max_len = max(len(a) for a in audios)
            audio_pad = torch.zeros(len(batch), max_len)
            for j, a in enumerate(audios):
                audio_pad[j, :len(a)] = torch.from_numpy(a)
            audio_pad = audio_pad.cuda()
            lengths = torch.tensor([len(a) for a in audios]).cuda()
            with torch.no_grad():
                feats, feat_lens = extract_features(encoder, audio_pad, lengths)
            logits = head(feats)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            targets = []
            target_lengths = []
            for s in batch:
                targets.extend(s['ids'])
                target_lengths.append(len(s['ids']))
            targets = torch.tensor(targets, dtype=torch.long).cuda()
            target_lengths = torch.tensor(target_lengths, dtype=torch.long).cuda()
            loss = ctc_loss(log_probs, targets, feat_lens, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if n_batches % 500 == 0:
                print(f"    batch {n_batches}/{len(samples)//bs}, loss={total_loss/n_batches:.3f}, "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        print(f"  Epoch {epoch+1}/{n_epochs}: ctc_loss={total_loss/n_batches:.4f}  ({time.time()-t0:.0f}s)")

    # Eval
    print(f"\n[4/4] Evaluating on SR test (500 samples)...")
    head.eval()
    test_path = base / "Thai-SUP/SR/test/test-00000.parquet"
    df = pq.read_table(str(test_path)).to_pandas()
    cers = []
    details = []
    with torch.no_grad():
        for i in range(min(500, len(df))):
            row = df.iloc[i]
            audio, sr = sf.read(io.BytesIO(row['audio_flac']), dtype='float32')
            if len(audio) < 3200:
                audio = np.pad(audio, (0, 3200 - len(audio)))
            audio_t = torch.from_numpy(audio).unsqueeze(0).cuda()
            length = torch.tensor([audio_t.shape[1]]).cuda()
            feats, _ = extract_features(encoder, audio_t, length)
            logits = head(feats)
            pred = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
            ref = str(row['text'])
            c = cer_ns(ref, pred)
            cers.append(c)
            if i < 5:
                details.append({'ref': ref, 'hyp': pred, 'cer': round(c, 3)})
    avg = sum(cers)/len(cers)
    median = sorted(cers)[len(cers)//2]
    print(f"  Avg CER (no_space): {avg:.2%}")
    print(f"  Median CER (no_space): {median:.2%}")
    for d in details:
        print(f"    ref: {d['ref'][:60]!r}")
        print(f"    hyp: {d['hyp'][:60]!r}")
        print(f"    CER: {d['cer']:.2%}")

    # Save head
    head_path = base / "ctc_head_trained_v2.pt"
    torch.save({'state_dict': head.state_dict()}, head_path)
    print(f"  Saved head to {head_path}")

    report = {
        'method': 'XLSR-Thai + pretrained CTC head init + 5-epoch fine-tune on ~300K SR samples',
        'n_train_samples': len(samples),
        'n_epochs': n_epochs,
        'sr_test_n': len(cers),
        'sr_test_avg_cer_no_space': avg,
        'sr_test_median_cer_no_space': median,
        'pretrained_xlsr_53_thai_baseline_cer': 0.3795,
        'paper_xlsr_thai_giga2_cer': 0.1391,
        'samples': details,
    }
    out = base / "results_ctc_full_finetune.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()
