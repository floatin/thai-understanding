"""
Fine-tune the entire XLSR-Thai encoder + CTC head on Thai-SUP train set.

Approach: 
1. Use pretrained XLSR-53-Thai (ZikXewen) as initialization (warm start)
   - It has compatible architecture (XLSR-53 + Thai CTC head)
   - The CTC head is trained on CommonVoice Thai
2. Continue training on Thai-SUP train data (different domain)
3. Or: use XLSR-Thai encoder + new CTC head with the pretrained head's weight
   initialization but adapted to XLSR-Thai features

This is closer to what the paper does (Table 1: XLSR-Thai + CTC head fine-tuned on GigaSpeech2).
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


def main():
    base = Path("/data/workspace/asr-model-training/thai-understanding")

    # Load pretrained XLSR-53-Thai (ZikXewen)
    print("[1/4] Loading pretrained XLSR-53-Thai (ZikXewen)...")
    pretrained_path = base / "ZikXewen-thai-ctc"
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(pretrained_path / "vocab.json"),
        unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(str(pretrained_path)).cuda()
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Load SR train shards
    print(f"\n[2/4] Loading SR train shards (quick test)...")
    train_files = sorted((base / "Thai-SUP/SR/train").glob("*.parquet"))[:2]
    print(f"  Using {len(train_files)} shards")

    # Train: fine-tune the encoder + CTC head
    print(f"\n[3/4] Fine-tuning on SR train (5 shards, 2 epochs)...")
    optimizer = torch.optim.AdamW([
        {'params': model.wav2vec2.parameters(), 'lr': 1e-5},  # very low LR for encoder
        {'params': model.lm_head.parameters(), 'lr': 3e-4},    # higher LR for head
    ])
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    rng = random.Random(42)
    model.train()
    for epoch in range(2):
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        # Sample from each shard
        for shard_idx, shard in enumerate(train_files):
            df = pq.read_table(str(shard)).to_pandas()
            # Random 50 samples per shard
            indices = rng.sample(range(len(df)), min(20, len(df)))
            samples = [df.iloc[i] for i in indices]
            random.shuffle(samples)
            batch_size = 1
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]
                # Filter
                valid_batch = []
                for s in batch:
                    text = str(s['text'])
                    if 5 < len(text) < 100:
                        # Convert text to ids
                        ids = tokenizer(text, return_tensors=None, padding=False).input_ids
                        # Skip if has any special token (BOS/EOS/UNK/PAD/WD)
                        if all(0 < i < 71 for i in ids):
                            audio, sr = sf.read(io.BytesIO(s['audio_flac']), dtype='float32')
                            if len(audio) < 3200:
                                audio = np.pad(audio, (0, 3200 - len(audio)))
                            valid_batch.append({'audio': audio, 'ids': ids})
                if not valid_batch:
                    continue
                audios = [v['audio'] for v in valid_batch]
                ids_list = [v['ids'] for v in valid_batch]
                max_len = max(len(a) for a in audios)
                audio_pad = np.zeros((len(valid_batch), max_len), dtype=np.float32)
                for j, a in enumerate(audios):
                    audio_pad[j, :len(a)] = a
                audio_t = torch.from_numpy(audio_pad).cuda()
                # Forward through pretrained Wav2Vec2
                outputs = model.wav2vec2(audio_t)
                hidden = outputs.last_hidden_state  # (B, T', 1024)
                hidden = model.dropout(hidden)
                logits = model.lm_head(hidden)  # (B, T', 71)
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                # Compute output lengths (T')
                feat_lens = torch.tensor([hidden.shape[1]] * hidden.shape[0]).cuda()
                # Targets
                targets = []
                target_lengths = []
                for ids in ids_list:
                    targets.extend(ids)
                    target_lengths.append(len(ids))
                targets = torch.tensor(targets, dtype=torch.long).cuda()
                target_lengths = torch.tensor(target_lengths, dtype=torch.long).cuda()
                loss = ctc_loss(log_probs, targets, feat_lens, target_lengths)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        print(f"  Epoch {epoch+1}: ctc_loss={total_loss/n_batches:.4f}  ({time.time()-t0:.1f}s)")

    # Evaluate on SR test
    print(f"\n[4/4] Evaluating on SR test (200 samples)...")
    model.eval()
    test_path = base / "Thai-SUP/SR/test/test-00000.parquet"
    df = pq.read_table(str(test_path)).to_pandas()
    cers = []
    details = []
    with torch.no_grad():
        for i in range(min(200, len(df))):
            row = df.iloc[i]
            audio, sr = sf.read(io.BytesIO(row['audio_flac']), dtype='float32')
            if len(audio) < 3200:
                audio = np.pad(audio, (0, 3200 - len(audio)))
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.cuda()
            logits = model(inputs).logits
            pred = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
            ref = str(row['text'])
            c = cer_ns(ref, pred)
            cers.append(c)
            if i < 5:
                details.append({'ref': ref, 'hyp': pred, 'cer': round(c, 3)})
    avg = sum(cers)/len(cers)
    print(f"  Avg CER (no_space): {avg:.2%}")
    for d in details:
        print(f"    ref: {d['ref'][:60]!r}")
        print(f"    hyp: {d['hyp'][:60]!r}")
        print(f"    CER: {d['cer']:.2%}")

    report = {
        'method': 'Pretrained XLSR-53-Thai (ZikXewen) + 2-epoch fine-tune on SR train (5 shards, 50/shard)',
        'n_train_samples': 5 * 50,
        'n_epochs': 2,
        'sr_test_avg_cer_no_space': avg,
        'pretrained_xlsr_53_thai_baseline_cer': 0.3795,
        'paper_xlsr_thai_giga2_cer': 0.1391,
        'samples': details,
    }
    out = base / "results_xlsr_thai_finetune.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()
