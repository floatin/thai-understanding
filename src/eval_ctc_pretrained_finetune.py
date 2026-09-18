"""
Fine-tune ONLY the CTC head (encoder frozen) using pretrained head as init.
This is the closest "warm-start" we can do given GPU memory constraints.
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

    # Load XLSR-Thai encoder
    print("[1/4] Loading XLSR-Thai encoder on GPU...")
    encoder, cfg = load_xlsr_thai(str(base / "XLSR-Thai/checkpoint_best.pt"))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Load pretrained CTC head weights
    print(f"[2/4] Loading pretrained CTC head from ZikXewen XLSR-53-Thai...")
    pretrained = Wav2Vec2ForCTC.from_pretrained(str(base / "ZikXewen-thai-ctc"))
    pretrained_lm_head_w = pretrained.lm_head.weight.detach().clone()  # (71, 1024)
    pretrained_lm_head_b = pretrained.lm_head.bias.detach().clone()  # (71,)
    del pretrained
    print(f"  CTC head: {pretrained_lm_head_w.shape}, dtype={pretrained_lm_head_w.dtype}")
    
    # Create CTC head with pretrained init
    # Pad to a larger vocab to handle the XLSR-Thai's wider char distribution
    n_chars = 73  # pretrained uses 73 outputs (with special tokens)
    head = nn.Linear(1024, n_chars).cuda()
    # Initialize: copy pretrained weights where chars overlap, random for new chars
    with torch.no_grad():
        # Use pretrained weights directly (they are for XLSR-53 features, may not work for XLSR-Thai but start there)
        head.weight.copy_(pretrained_lm_head_w.cuda())
        head.bias.copy_(pretrained_lm_head_b.cuda())
    head.train()

    # Processor (for inference)
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(base / "ZikXewen-thai-ctc/vocab.json"),
        unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Load SR train shards
    print(f"\n[3/4] Loading SR train (3 shards, 100 samples each, 3 epochs)...")
    train_files = sorted((base / "Thai-SUP/SR/train").glob("*.parquet"))[:3]
    rng = random.Random(42)
    samples = []
    for f in train_files:
        df = pq.read_table(str(f)).to_pandas()
        indices = rng.sample(range(len(df)), min(100, len(df)))
        for i in indices:
            row = df.iloc[i]
            text = str(row['text'])
            ids = tokenizer(text, return_tensors=None, padding=False).input_ids
            if 5 < len(text) < 100 and all(0 < i < 71 for i in ids):
                samples.append({'audio_flac': row['audio_flac'], 'ids': ids, 'text': text})
    print(f"  Total: {len(samples)} samples")

    # Train
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    for epoch in range(3):
        random.shuffle(samples)
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(samples), 4):  # batch_size=4
            batch = samples[i:i+4]
            # Load audio
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
            # Encode
            with torch.no_grad():
                feats, feat_lens = extract_features(encoder, audio_pad, lengths)
            # CTC head
            logits = head(feats)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            # Targets
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
        print(f"  Epoch {epoch+1}: ctc_loss={total_loss/n_batches:.4f}  ({time.time()-t0:.1f}s)")

    # Eval on SR test
    print(f"\n[4/4] Evaluating on SR test (200 samples)...")
    head.eval()
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
    print(f"  Avg CER (no_space): {avg:.2%}")
    for d in details:
        print(f"    ref: {d['ref'][:60]!r}")
        print(f"    hyp: {d['hyp'][:60]!r}")
        print(f"    CER: {d['cer']:.2%}")

    report = {
        'method': 'XLSR-Thai encoder + pretrained CTC head (warm init) + 3-epoch fine-tune on SR train (300 samples)',
        'n_train_samples': len(samples),
        'n_epochs': 3,
        'sr_test_avg_cer_no_space': avg,
        'pretrained_xlsr_53_thai_baseline_cer': 0.3795,
        'paper_xlsr_thai_giga2_cer': 0.1391,
        'samples': details,
    }
    out = base / "results_ctc_finetune.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()
